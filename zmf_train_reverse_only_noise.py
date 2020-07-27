# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob

input_dir = '../../../../../media/zhangmf/C14D581BDA18EBFA/data/Sony/Sony/long/'
gt_dir = '../../../../../media/zhangmf/C14D581BDA18EBFA/data/Sony/Sony/short/'
checkpoint_dir = '../../workplace/4-learn-dark/converse-noise-result/'
result_dir = '../../workplace/4-learn-dark/converse-noise-result/'

# get train IDs
train_fns = glob.glob(input_dir + '0*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]
print(train_ids)

ps = 1024  # patch size for training
save_freq = 500

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    train_ids = train_ids[120:140]

def lrelu(x):
    return tf.maximum(x * 0.2, x)

def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output

def network(input):
    # input = tf.space_to_depth(input, 2) # H*W*3 -> H/2*W/2*12
    # H/2*W/2*4
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512)
    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)
    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 4, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
    
    return conv10 # H/2*W/2*4

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :], # r
                          im[0:H:2, 1:W:2, :], # g
                          im[1:H:2, 1:W:2, :], # b
                          im[1:H:2, 0:W:2, :]), axis=2) # g
    return out

def seeraw(some_patch):
    # (1, 1024, 1024, 4) -> (1024,1024,3)
    raw = some_patch[0,:,:,:]
    H = raw.shape[0]
    W = raw.shape[1]

    r = raw[:,:,0]
    g = (raw[:,:,1] + raw[:,:,3])/2.0
    b = raw[:,:,2]

    seeimg = np.zeros([H,W,3])
    seeimg[:,:,0] = r
    seeimg[:,:,1] = g
    seeimg[:,:,2] = b

    return seeimg

def smallrgb(bigrgb):
    # (1, 2048, 2048, 3) -> (1024,1024,3)
    smimg = bigrgb[0,0::2,0::2,:]
    return smimg

sess = tf.Session()
in_noise_zmf = tf.placeholder(tf.float32, [None, None, None, 4])  # simple noise based on bright raw, H/2*W/2*4
gt_noise_zmf = tf.placeholder(tf.float32, [None, None, None, 4])  # dark raw * ratio - bright raw, H/2*W/2*4
out_noise_zmf = network(in_noise_zmf) # raw output img, H/2*W/2*4

# zmf: 4 image losses
G_loss = tf.reduce_mean(tf.abs(out_noise_zmf - gt_noise_zmf))
# G_loss = tf.reduce_mean(tf.square(out_noise_zmf - gt_noise_zmf))
# G_loss = 1. - tf.image.ssim(out_noise_zmf, gt_noise_zmf, 1.0)
# G_loss = 1. - tf.image.ssim_multiscale(out_noise_zmf, gt_noise_zmf , 1.0)

t_vars = tf.trainable_variables()
lr = tf.placeholder(tf.float32)
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

# Raw data takes long time to load. Keep them in memory after loaded.
clean_raws = [None] * 6000
noisy_raws = {}
noisy_raws['300'] = [None] * len(train_ids)
noisy_raws['250'] = [None] * len(train_ids)
noisy_raws['100'] = [None] * len(train_ids)

g_loss = np.zeros((5000, 1))

allfolders = glob.glob(result_dir + '*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4
for epoch in range(lastepoch, 4001):
    if os.path.isdir(result_dir + '%04d' % epoch):
        continue
    cnt = 0
    if epoch > 2000:
        learning_rate = 1e-5
    et = time.time()

    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        clean_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
        clean_path = clean_files[0] # only one 
        clean_fn = os.path.basename(clean_path)
        # print(clean_fn)

        noisy_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
        noisy_path = noisy_files[np.random.randint(0, len(noisy_files))] # 0.1 and 0.04 and sometimes 0.033, pick one
        noisy_fn = os.path.basename(noisy_path)



        bright_exposure = float(clean_fn[9:-5])
        dark_exposure = float(noisy_fn[9:-5])
        ratio = min(bright_exposure / dark_exposure, 300)

        st = time.time()
        cnt += 1

        if noisy_raws[str(ratio)[0:3]][ind] is None:
            clean_raw = rawpy.imread(clean_path)
            clean_raws[ind] = np.expand_dims(pack_raw(clean_raw), axis=0)
            # print(clean_raws[ind].shape) # (1, 2848, 4256, 3)

            noisy_raw = rawpy.imread(noisy_path)
            noisy_raws[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(noisy_raw), axis=0) * ratio
            # print(noisy_raws[str(ratio)[0:3]][ind].shape) # (1, 1424, 2128, 4)
            # print("ratio = " + str(ratio))

        # crop
        H = noisy_raws[str(ratio)[0:3]][ind].shape[1] # H/2
        W = noisy_raws[str(ratio)[0:3]][ind].shape[2] # W/2

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        noisy_patch = noisy_raws[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        clean_patch = clean_raws[ind][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = noisy_patch - clean_patch
        a = 0.1
        b = 0.1
        input_patch =  np.random.normal(0,a*clean_patch+b)
        # print(gt_patch.shape, input_patch.shape) # (1, 1024, 1024, 4) (1, 1024, 1024, 4)

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        gt_patch = np.maximum(np.minimum(gt_patch*3.0, 1.0),0.0)
        input_patch = np.maximum(np.minimum(input_patch*3.0, 1.0),0.0)
        # scipy.misc.toimage(seeraw(gt_patch) * 255, high=255, low=0, cmin=0, cmax=255).save(result_dir + 'try.jpg')

        _, G_current, output = sess.run([G_opt, G_loss, out_noise_zmf],
                                        feed_dict={in_noise_zmf: input_patch, gt_noise_zmf: gt_patch, lr: learning_rate})
        output = np.minimum(np.maximum(output, 0), 1)
        g_loss[ind] = G_current

        print("%d %d Loss=%.3f Time=%.3f,%.3f" % (epoch, cnt, np.mean(g_loss[np.where(g_loss)]), time.time() - st, time.time() - et))

        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)

            temp = np.concatenate((seeraw(noisy_patch),seeraw(clean_patch),seeraw(gt_patch),seeraw(output),seeraw(input_patch),seeraw(output+clean_patch)), axis=1)
            scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))

    saver.save(sess, checkpoint_dir + 'model.ckpt')
