# version 20200728
# import glob
# import numpy as np
# import matplotlib.pyplot as plt
# import rawpy
# import os
# import cv2

# from scanf import scanf
# from tqdm import tqdm

# GENERAL
def get_such_addrs(addr_pattern):
    import glob
    """
    input a pattern of file address like './*.JPG'\n
    return an array of all these files' address
    """
    a = glob.glob(addr_pattern)
    a.sort()
    return a

def path_exists(addr):
    """
    input a path of a file or folder\n
    return True if it exists, else False
    """
    import os
    return os.path.exists(addr)

def mkdir(addr, ask=1):
    """
    input a addr of new folder\n
    ask: 0 - no asking, try creating folder once\n
    ask:-1 - no asking, if the folder exist then empty it (dangerous)\n
    ask: 1 - ask if the folder exist
    """
    import os
    assert(ask in [-1,0,1])
    if not path_exists(addr):
        os.makedirs(addr)
        cprint('Created a folder at '+addr)
    else:
        print('The folder '+addr+' already exists')
        if not ask == 0:
            yn = 'n'
            if ask == 1:
                yn = input("delete it and create an empty one? (y/n)")
                assert(yn in ['y','n'])
            if yn == 'y' or ask == -1:
                os.removedirs(addr)
                os.makedirs(addr)

def clean_dst_file(dst_file):
    """
    input a file addr\n
    Create the output folder, if necessary\n
    empty the output folder of previous predictions, if any
    """
    # Create the output folder, if necessary
    import os
    dst_file_dir = os.path.dirname(dst_file)
    if not os.path.exists(dst_file_dir):
        os.makedirs(dst_file_dir)

    # Empty the output folder of previous predictions, if any
    if os.path.exists(dst_file):
        os.remove(dst_file)

def shell_cmd(c):
    """
    input a shell command c\n
    execute it in shell\n
    return 0 if success, else return 256
    """
    import os
    return os.system(c)

def python_code(c):
    """
    input a string of python code\n
    and execute it
    """
    exec(c)

def python_expr(e):
    """
    input a string of python expression\n
    return the expression's value
    """
    return eval(e)

def basename(addr):
    """
    input a full address\n
    return the base file name
    """
    import os
    return os.path.basename(addr)

def cprint(s,color=-1):
    """
    input a something s\n
    print it in color 0-red, 1-green, 2-blue, default-yellow
    """
    if color == 0: print('\033[31m',s,'\033[0m')
    elif color == 1: print('\033[32m',s,'\033[0m')
    elif color == 2: print('\033[34m',s,'\033[0m')
    else: print('\033[33m',s,'\033[0m')

# DEAL WITH IMAGES
def iminfo(im):
    """
    input an im array\n
    print the im shape, min, mean, max
    """
    import numpy as np
    print('\033[33m')
    print('shape: ', im.shape)
    print('min: %f, mean: %f, max: %f' % (np.min(im),np.mean(im),np.max(im)))
    print('\033[0m')

def imread(addr):
    """
    input a string of im address\n
    return an im array (if RGB then 0R,1G,2B)
    """
    import matplotlib.pyplot as plt
    im = plt.imread(addr)
    return im

def imsave(addr,im):
    """
    input a string of save address, an im array\n
    save the image to the address
    """
    import matplotlib.pyplot as plt
    return plt.imsave(addr,im)

def imshow(im):
    """
    input an im array\n
    show the image directly
    """
    import matplotlib.pyplot as plt
    plt.imshow(im)
    plt.show()

def imscale(im, to_mean, print_ratio=False):
    """
    input image [0,1], the aiming mean\n
    if print_ratio then do so\n
    return the scaled image
    """
    import numpy as np
    assert(np.max(im)<=1)
    ratio = to_mean/np.mean(im)
    im = im*ratio
    im = np.maximum(im,0.)
    im = np.minimum(im,1.)
    if print_ratio: print('ratio =',ratio)
    return im

def imcrop(im,to_h,to_w,skip_n=1,center=False):
    """
    input big image, crop to h,w, skip every n pixels, crop in the center or not\n
    return a cropped image
    """
    assert(im.shape[0]>=to_h)
    assert(im.shape[1]>=to_w)
    if center == False:
        if im.ndim == 3:
            newim = im[0:to_h*skip_n:skip_n,0:to_w*skip_n:skip_n,:]
        elif im.ndim == 2:
            newim = im[0:to_h*skip_n:skip_n,0:to_w*skip_n:skip_n]
    else:
        marg_h = im.shape[0] - to_h
        marg_w = im.shape[1] - to_w
        h1=h2=w1=w2=0
        h1 = marg_h/2 if marg_h%2==0 else marg_h/2-0.5
        h2 = marg_h/2 if marg_h%2==0 else marg_h/2+0.5
        w1 = marg_w/2 if marg_w%2==0 else marg_w/2-0.5
        w2 = marg_w/2 if marg_w%2==0 else marg_w/2+0.5
        h1,h2,w1,w2 = int(h1),int(h2),int(w1),int(w2)
        newim = im[h1:-h2:skip_n,w1:-w2:skip_n,:]
        if im.ndim == 3:
            newim = im[h1:-h2:skip_n,w1:-w2:skip_n,:]
        elif im.ndim == 2:
            newim = im[h1:-h2:skip_n,w1:-w2:skip_n]
    return newim

def imzoom(img, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    Based on:
        - Scipy rotate and zoom an image without changing its dimensions
        https://stackoverflow.com/a/48097478
        Written by Mohamed Ezz
        License: MIT License
    """
    import numpy as np
    import cv2
    height, width = img.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])

    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0, 0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')

    assert(result.shape[0] == height and result.shape[1] == width)
    return result

# DEAL WITH RAW IMAGES
"""example
    # Method 1 (direct)
    im = raw_process('./a.ARW', wb=2, mywb=[2,1,2,1], auto_bright=False, output_bps=16, demosaic=True)

    # Method 2 (explicitly)
    im2d = raw_read('./a.ARW',2047,16383)
    im4d = pack_bayer(im2d)
    # note that compared to jpeg files raw files have margin
    im = nd_to_3d(im4d,[0],[1,3],[2],.5)
    im = imscale(im,0.4)
"""

def raw_info(addr):
    """
    input a string of raw address\n
    print raw_pattern, black_level, shot_wb, shape, min, mean, max
    """
    import numpy as np
    import rawpy
    raw = rawpy.imread(addr)
    im = raw.raw_image_visible.astype(np.float32)
    print('\033[33m')
    print('raw pattern (maybe):')
    print(raw.color_desc)
    print(raw.raw_pattern)
    print('black level (maybe): ', raw.black_level_per_channel)
    print('shot white balance: ', raw.camera_whitebalance)
    print('shape: ', im.shape)
    print('min: %f, mean: %f, max: %f' % (np.min(im),np.mean(im),np.max(im)))
    print('\033[0m')

def raw_read(addr, rmin=2047, rmax=16383, orig=False):
    """
    input a string of raw address\n
    consult `raw_info()` before you decide the min and max\n
    min should be slightly bigger than np.min(all_im), always the black level will work\n
    max should be slightly bigger than np.max(all_im), always 16383 will work\n
    try differnet values until the contrast is fine\n
    if orig then return the original im array [0,2^n]\n
    else return (im2d-min)/max [0,1]
    """
    import numpy as np
    import rawpy
    raw = rawpy.imread(addr)
    im = raw.raw_image_visible.astype(np.float32)
    if orig: return im

    im = im - rmin
    im = im / (rmax-rmin)

    im = np.maximum(im,0)
    im = np.minimum(im,1)
    return im

def pack_bayer(bayer_2d):
    """
    input an 2d image (H,W) from a bayer raw\n
    return a 4d image (H/2,W/2,4)\n
    a b\n
    c d\n
    im[x,x,0]-a, im[x,x,1]-b, im[x,x,2]-d, im[x,x,3]-c\n
    
    normally,\n
    im[x,x,0]-R, im[x,x,1]-G, im[x,x,2]-B, im[x,x,3]-G
    """
    import numpy as np
    im = bayer_2d
    im = np.expand_dims(im, axis=2)
    
    out = np.concatenate((im[0::2, 0::2, :], # a
                          im[0::2, 1::2, :], # b
                          im[1::2, 1::2, :], # d
                          im[1::2, 0::2, :]  # c
                          ), axis=2)
    return out

def pack_XTrans_au(bayer_2d):
    """
    input a 2d xtrans raw image, pattern like this:\n
    [1, 1, 0, 1, 1, 2],\n
    [1, 1, 2, 1, 1, 0],\n
    [2, 0, 1, 0, 2, 1],\n
    [1, 1, 2, 1, 1, 0],\n
    [1, 1, 0, 1, 1, 2],\n
    [0, 2, 1, 2, 0, 1]\n
    return a 9d image\n
    R-0,4d G-1,5,6,7,8d B-1,5d\n
    """
    import numpy as np
    im = bayer_2d

    img_shape = im.shape
    H = (img_shape[0] // 6) * 6
    W = (img_shape[1] // 6) * 6

    out = np.zeros((H // 3, W // 3, 9))

    # 0 R
    out[0::2, 0::2, 0] = im[0:H:6, 0:W:6]
    out[0::2, 1::2, 0] = im[0:H:6, 4:W:6]
    out[1::2, 0::2, 0] = im[3:H:6, 1:W:6]
    out[1::2, 1::2, 0] = im[3:H:6, 3:W:6]

    # 1 G
    out[0::2, 0::2, 1] = im[0:H:6, 2:W:6]
    out[0::2, 1::2, 1] = im[0:H:6, 5:W:6]
    out[1::2, 0::2, 1] = im[3:H:6, 2:W:6]
    out[1::2, 1::2, 1] = im[3:H:6, 5:W:6]

    # 1 B
    out[0::2, 0::2, 2] = im[0:H:6, 1:W:6]
    out[0::2, 1::2, 2] = im[0:H:6, 3:W:6]
    out[1::2, 0::2, 2] = im[3:H:6, 0:W:6]
    out[1::2, 1::2, 2] = im[3:H:6, 4:W:6]

    # 4 R
    out[0::2, 0::2, 3] = im[1:H:6, 2:W:6]
    out[0::2, 1::2, 3] = im[2:H:6, 5:W:6]
    out[1::2, 0::2, 3] = im[5:H:6, 2:W:6]
    out[1::2, 1::2, 3] = im[4:H:6, 5:W:6]

    # 5 B
    out[0::2, 0::2, 4] = im[2:H:6, 2:W:6]
    out[0::2, 1::2, 4] = im[1:H:6, 5:W:6]
    out[1::2, 0::2, 4] = im[4:H:6, 2:W:6]
    out[1::2, 1::2, 4] = im[5:H:6, 5:W:6]

    out[:, :, 5] = im[1:H:3, 0:W:3]
    out[:, :, 6] = im[1:H:3, 1:W:3]
    out[:, :, 7] = im[2:H:3, 0:W:3]
    out[:, :, 8] = im[2:H:3, 1:W:3]
    return out

def pack_XTrans_me(bayer_2d):
    """
    input a 2d xtrans raw image, pattern like this:\n
    [0, 2, 1, 2, 0, 1],\n
    [1, 1, 0, 1, 1, 2],\n
    [1, 1, 2, 1, 1, 0],\n
    [2, 0, 1, 0, 2, 1],\n
    [1, 1, 2, 1, 1, 0],\n
    [1, 1, 0, 1, 1, 2]\n
    return a 9d image\n
    R-0,4d G-1,5,6,7,8d B-1,5d\n
    """
    import numpy as np
    im = bayer_2d

    img_shape = im.shape
    H = (img_shape[0] // 6) * 6
    W = (img_shape[1] // 6) * 6

    out = np.zeros((H // 3, W // 3, 9))

    # 0 R
    out[0::2, 0::2, 0] = im[5:H:6, 0:W:6]
    out[0::2, 1::2, 0] = im[5:H:6, 4:W:6]
    out[1::2, 0::2, 0] = im[2:H:6, 1:W:6]
    out[1::2, 1::2, 0] = im[2:H:6, 3:W:6]

    # 1 G
    out[0::2, 0::2, 1] = im[5:H:6, 2:W:6]
    out[0::2, 1::2, 1] = im[5:H:6, 5:W:6]
    out[1::2, 0::2, 1] = im[2:H:6, 2:W:6]
    out[1::2, 1::2, 1] = im[2:H:6, 5:W:6]

    # 1 B
    out[0::2, 0::2, 2] = im[5:H:6, 1:W:6]
    out[0::2, 1::2, 2] = im[5:H:6, 3:W:6]
    out[1::2, 0::2, 2] = im[2:H:6, 0:W:6]
    out[1::2, 1::2, 2] = im[2:H:6, 4:W:6]

    # 4 R
    out[0::2, 0::2, 3] = im[0:H:6, 2:W:6]
    out[0::2, 1::2, 3] = im[1:H:6, 5:W:6]
    out[1::2, 0::2, 3] = im[4:H:6, 2:W:6]
    out[1::2, 1::2, 3] = im[3:H:6, 5:W:6]

    # 5 B
    out[0::2, 0::2, 4] = im[1:H:6, 2:W:6]
    out[0::2, 1::2, 4] = im[0:H:6, 5:W:6]
    out[1::2, 0::2, 4] = im[3:H:6, 2:W:6]
    out[1::2, 1::2, 4] = im[4:H:6, 5:W:6]

    out[:, :, 5] = im[0:H:3, 0:W:3]
    out[:, :, 6] = im[0:H:3, 1:W:3]
    out[:, :, 7] = im[1:H:3, 0:W:3]
    out[:, :, 8] = im[1:H:3, 1:W:3]
    return out

def nd_to_3d(ndarray, Rlist, Glist, Blist, G_plus=0):
    """
    input an nd image, Rchannels, Gchannels, Bchannels\n
    G_plus rise, green color down
    return a 3d image
    """
    import numpy as np
    out = np.zeros([ndarray.shape[0],ndarray.shape[1],3])

    for r in Rlist: out[:,:,0] += ndarray[:,:,r]
    out[:,:,0] /= len(Rlist)
    for g in Glist: out[:,:,1] += ndarray[:,:,g]
    out[:,:,1] /= (len(Glist)+G_plus)
    for b in Blist: out[:,:,2] += ndarray[:,:,b]
    out[:,:,2] /= len(Blist)

    return out

def raw_process(addr, wb=0, mywb=[2,1,2,1], auto_bright=False, output_bps=16, demosaic=True):
    """
    input a raw address (raw - H*W*2),\n
    wb = 0-use_camera_wb, 1-use_auto_wb, 2-use_mywb,\n
    if use_mywb, you can consult `raw_info` for wb values
    if auto_bright then auto enhance the dark images,\n
    output_bps = 8 or 16,\n
    if demosaic then return image H*W*3, else H/2*W/2*3
    """
    import rawpy
    import numpy as np
    assert(wb in [0,1,2])
    assert(output_bps in [8,16])
    cameraWb = True
    autoWb = False
    userWb = None
    if wb == 1:
        cameraWb = False
        autoWb = True
    if wb == 2:
        cameraWb = False
        userWb = mywb
    halfSize = False if demosaic else True
    noAutoBright = False if auto_bright else True
    outputBps = output_bps

    raw = rawpy.imread(addr)
    im = raw.postprocess(use_camera_wb=cameraWb, use_auto_wb=autoWb, user_wb=userWb, half_size=halfSize, no_auto_bright=noAutoBright, output_bps=outputBps)
    if outputBps == 16:
        im = np.float32(im/65535.0)
    else:
        im = np.float32(im/255.0)
    return im

# OPTICAL FLOW
TAG_FLOAT = 202021.25

def flow_read(src_file):
    """Read optical flow stored in a .flo, .pfm, or .png file
    Args:
        src_file: Path to flow file
    Returns:
        flow: optical flow in [h, w, 2] format
    Refs:
        - Interpret bytes as packed binary data
        Per https://docs.python.org/3/library/struct.html#format-characters:
        format: f -> C Type: float, Python type: float, Standard size: 4
        format: d -> C Type: double, Python type: float, Standard size: 8
    Based on:
        - To read optical flow data from 16-bit PNG file:
        https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py
        Written by Clément Pinard, Copyright (c) 2017 Clément Pinard
        MIT License
        - To read optical flow data from PFM file:
        https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/pfm.py
        Written by Ruoteng Li, Copyright (c) 2017 Ruoteng Li
        License Unknown
        - To read optical flow data from FLO file:
        https://github.com/daigo0927/PWC-Net_tf/blob/master/flow_utils.py
        Written by Daigo Hirooka, Copyright (c) 2018 Daigo Hirooka
        MIT License
    """
    import numpy as np
    import os
    import cv2
    # Read in the entire file, if it exists
    assert(os.path.exists(src_file))

    if src_file.lower().endswith('.flo'):

        with open(src_file, 'rb') as f:

            # Parse .flo file header
            tag = float(np.fromfile(f, np.float32, count=1)[0])
            assert(tag == TAG_FLOAT)
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]

            # Read in flow data and reshape it
            flow = np.fromfile(f, np.float32, count=h * w * 2)
            flow.resize((h, w, 2))

    elif src_file.lower().endswith('.png'):

        # Read in .png file
        flow_raw = cv2.imread(src_file, -1)

        # Convert from [H,W,1] 16bit to [H,W,2] float formet
        flow = flow_raw[:, :, 2:0:-1].astype(np.float32)
        flow = flow - 32768
        flow = flow / 64

        # Clip flow values
        flow[np.abs(flow) < 1e-10] = 1e-10

        # Remove invalid flow values
        invalid = (flow_raw[:, :, 0] == 0)
        flow[invalid, :] = 0

    elif src_file.lower().endswith('.pfm'):

        with open(src_file, 'rb') as f:

            # Parse .pfm file header
            tag = f.readline().rstrip().decode("utf-8")
            assert(tag == 'PF')
            dims = f.readline().rstrip().decode("utf-8")
            w, h = map(int, dims.split(' '))
            scale = float(f.readline().rstrip().decode("utf-8"))

            # Read in flow data and reshape it
            flow = np.fromfile(f, '<f') if scale < 0 else np.fromfile(f, '>f')
            flow = np.reshape(flow, (h, w, 3))[:, :, 0:2]
            flow = np.flipud(flow)
    else:
        raise IOError

    return flow

def flow_write(flow, dst_file):
    """Write optical flow to a .flo file
    Args:
        flow: optical flow
        dst_file: Path where to write optical flow
    """
    import numpy as np
    # Create the output folder, if necessary
    # Empty the output folder of previous predictions, if any
    clean_dst_file(dst_file)

    # Save optical flow to disk
    with open(dst_file, 'wb') as f:
        np.array(TAG_FLOAT, dtype=np.float32).tofile(f)
        height, width = flow.shape[:2]
        np.array(width, dtype=np.uint32).tofile(f)
        np.array(height, dtype=np.uint32).tofile(f)
        flow.astype(np.float32).tofile(f)

# STATISTICS

def normal(mean, std, size=1):
    """
    input mean value, standard deviation, output size\n
    return certain size of Gaussian Distribution result
    """
    import numpy as np
    return np.random.normal(mean,std,size)

def poisson(lam, size=1):
    """
    input lambda, output size\n
    ATTENTION: ONLY INTEGERS, if small number are used, please first multiply then divide\n
    return certain size of Poisson Distribution result
    """
    import numpy as np
    return np.random.poisson(lam,size)

def uniform(low, high, size=1):
    """
    input min value, max value, output size\n
    ATTENTION: [low,high)\n
    return certain size of Uniform Distribution result
    """
    import numpy as np
    return np.random.uniform(low,high,size)

def tukeylambda(lam_shape, loc=0, scale=1, size=1):
    """
    input: shape parameter lam_shape, middle location, scale, output size\n
    describe: A flexible distribution, able to represent and interpolate between the following distributions:\n
        Cauchy (lambda=−1)\n
        logistic (lambda=0)\n
        approx Normal (lambda=0.14)\n
        uniform from -1 to 1 (lambda=1)\n
    output: Random variates of certain size
    """
    from scipy.stats import tukeylambda
    return tukeylambda.rvs(lam_shape, loc, scale, size)

def see_distribution(see_dots,bin_num=1000,xmin=None,xmax=None,ymin=None,ymax=None):
    """
    input: one-dimensional array see_dots (eg. normal(0,1,size=1000), num of bins to collect dots, display xy maxmin )\n
    NOTE: if step effect is too strong, increase bin_num;\n 
    if the shape is not smooth, increase the len(see_dots)\n
    output: display what we want with 01Normal as ref
    """

    import matplotlib.pyplot as plt

    ref_dots = normal(0,1,size=5000000)

    # Make histograms
    plt.hist(ref_dots,bins=500,normed=True,histtype='step')
    plt.hist(see_dots,bins=bin_num,normed=True,histtype='step')     
     
    # Make a legend, set limits and show plot
    _ = plt.legend(('01Normal', 'see'))
    if xmin != None and xmax != None: plt.xlim(xmin, xmax)
    if ymin != None and ymax != None: plt.ylim(ymin, ymax)

    plt.show()
