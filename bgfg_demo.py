import time, math, sys
import cv2
import numpy as np
from bgfg_a import bgfg_a

def bgfg_side2side_compare(fp, fr, test_set=None, playback='per_frame', display='mask', output=None, display_size=(1680, 945)):
    '''
    test_set format:
        [('model', 'speed'), ...]
    playback: playback mode
        None(nonstop), 'auto', 'auto_slow', 'last', 'last_slow', 'per_frame'
    display: whether to draw mask/box on original image
        None(orig. img), 'bbox', 'bbox_nms', 'bbox_mask', bbox_nms_mask' 'mask', 'foreground'
    output: mode for saving display images
        None(no save), 'single', subplot', 'video'
    display_size: display size on screen
    '''
    # check empty
    if not test_set:
        print("Empty test set.")

    # set playback
    pause_time = 0.1 if playback.endswith('_slow') else 0.0

    cap = cv2.VideoCapture(fp)

    # set video output and 1st frame flag
    vid = None
    video_first_frame = True       

    #generate subplot dimensions
    w = math.ceil(math.sqrt(len(test_set)))
    h = math.ceil(len(test_set)/w)
    w_h_resize = (int(display_size[0]/w), int(display_size[1]/w))

    bgfgs = [bgfg_a(x[0], x[1]) for x in test_set]
    imgs = [None] * len(test_set)
    blank_image = np.zeros((w_h_resize[1], w_h_resize[0], 3), np.uint8)

    timers = [0.0] * len(test_set)

    for f in range(fr):
        _ret, im1 = cap.read()

        #generate foreground masked images (subplots)
        for j, bgfg in enumerate(bgfgs):
            #get foreground mask
            t0 = time.time()
            fgmask = bgfg.apply(im1)
            timers[j] += time.time() - t0

            if display == 'foreground':
                display_im = fgmask
            else:
                display_im = im1
                if display.endswith('mask'):
                    #create color foreground mask
                    display_im = bgfg.mask_fusion(display_im, fgmask)
                if display.startswith('bbox'):
                    #create foreground bounding boxes
                    display_im = bgfg.foreground_bbox(display_im, fgmask, True if display.find('nms') != -1 else False)

            if output == 'single': #save image
                cv2.imwrite('%05d_%s.jpg' % (f, test_set[j]), display_im)

            #resize image to subplot size
            imgs[j] = cv2.resize(display_im, w_h_resize)

        #arrange subplots
        img_h = [None] * h
        for i in range(h):
            img_w = [None] * w
            for j in range(w):
                idx = i*w+j
                if idx < len(imgs):
                    text = '%s %3.1ffps' % (test_set[idx], f/timers[idx])
                    img_w[j] = cv2.putText(imgs[idx], text, (1, 21), cv2.FONT_HERSHEY_PLAIN, 1.5, (150, 150, 150), 1, cv2.LINE_AA) #text shadow
                    img_w[j] = cv2.putText(img_w[j], text, (0, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    img_w[j] = blank_image
            img_h[i] = cv2.hconcat(img_w)
        img = cv2.vconcat(img_h)

        cv2.imshow('new', img)
        if output == 'subplot': #save image
            cv2.imwrite('%05d_subplot.jpg' % frame, img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        elif output == 'video':
            if video_first_frame:
                vid = cv2.VideoWriter("demo.avi", cv2.VideoWriter_fourcc(*'XVID'), math.ceil(cap.get(cv2.CAP_PROP_FPS)), (img.shape[1], img.shape[0]), True)
                video_first_frame = False
                # vid = cv2.VideoWriter("demo.avi", cv2.VideoWriter_fourcc(*'XVID'), math.ceil(cap.get(cv2.CAP_PROP_FPS)), display_size, True)
            vid.write(img)

        if playback.endswith('_frame'):
            cv2.waitKey()

        if cv2.waitKey(1) & 0xFF in [ord('q'), ord("Q"), 27]: #you can exit program any time by pressing Esc or Q/q
            cv2.destroyAllWindows()
            sys.exit()
        time.sleep(pause_time)

    if output == 'video':
        vid.release()

    print('Summary:')
    for i, timer in enumerate(timers):
        print('%s: %.1ffps' % (test_set[i], fr/timer))

    if playback.startswith('last'):
        cv2.waitKey()

def bgfg_test(fp, fr, test_set=['GMG', 'CNT', 'GSOC', 'LSBP', 'MOG', 'MOG2', 'KNN']):
    bgfg = bgfg_a('MOG2', '3')
    cap = cv2.VideoCapture(fp)
    for _i in range(fr):
        _ret, im1 = cap.read()
        cv2.imshow('', bgfg.apply_fusion(im1))
        cv2.waitKey()

if __name__ == "__main__":

    # src = "DJI_0002_15FPSa"
    # src = "20190527hd_p33a"
    src = "ivs104_20181109_1210a"
    filepath = 'F:/video/%s/%s.m4v' % (src, src)
    frame = 100
    models = ['MOG2', 'MOG', 'CNT', 'KNN']
    test_sets = [
       ('MOG2', 'normal'), ('KNN', 'normal'), ('CNT', 'normal'), ('MOG2', 'faster'), ('KNN', 'faster'), ('CNT', 'faster'), #three fast algo comarison
        # ('MOG2', 'pure'), ('KNN', 'pure'), ('CNT', 'pure'), ('MOG', 'pure'), ('GMG', 'pure'), ('GSOC', 'pure'), ('LSBP', 'pure'), #raw bgfg comparison
        # ('MOG2', 'slowest'), ('MOG2', 'slow'), ('MOG2', 'normal'), ('MOG2', 'fast'), ('MOG2', 'faster'), ('MOG2', 'fastest'), ('MOG2', 'flash'), ('MOG2', 'pure') #speed comparison
        # ('MOG', 'faster'), ('MOG2', 'faster'),  ('cuda_GMG', 'faster'), ('cuda_MOG', 'faster'), ('cuda_MOG2', 'faster'),  ('cuda_FGD', 'faster'), #CUDA comparison -- failed
        # ('MOG2', 'faster')
    ]

    if len(sys.argv) > 2: # has input arguments
        test_sets = [(sys.argv[1], sys.argv[2])]

    bgfg_side2side_compare(filepath, frame, test_sets, playback='last', display='mask', output='video')
    # bgfg_test(filepath, frame)
