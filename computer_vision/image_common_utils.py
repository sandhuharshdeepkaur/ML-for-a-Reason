#%% Get min (either of Width or Height), Max and Mean of size of image. Also get count of images
# for each tuple of dimensions
# image_dir = "./image_age/Train/"
def get_image_size_summary(image_dir):
    from keras.preprocessing import image

    # Get list of all files asuming that files are image type only
    #list_of_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    list_of_files = image.list_pictures(image_dir)
    #  arr_txt = [x for x in os.listdir() if x.endswith(".txt")]
    print("Count of image files read is ",  len(list_of_files))

    # Create a temporary DF to staore image sizes
    df = pd.DataFrame()

    # Iterate over each images and read their sizes
    for img_name in list_of_files: # [0:10] img_name = list_of_files[0]
        #img_path = os.path.join(image_dir, img_name)
        img_path = img_name
        img = image.load_img(img_path) # , target_size=(224, 224)
        df = pd.concat([df, pd.DataFrame({'W' : [img.size[0]], 'H' : [img.size[1]]})],axis=0) #top/bottom rbind
        del(img)

    # Get mIn, Max and Mean sizes
    size_min = np.min(df.min()); size_max = np.max(df.max()); size_mean = np.int(np.mean(df.mean()))
    print("Min, Max and Mean are ", size_min, size_max, size_mean)

    df = get_group_cat_many(df, df.columns.tolist())

    return({"size_min" : size_min, "size_max" : size_max, "size_mean" : size_mean,"summary" :df})

# Read the image
# W = img_summary['size_min']; H = W
def load_images(image_dir, W, H, image_array_divided_by = 1):
    from keras.preprocessing import image

    # Get list of all files asuming that files are image type only
    list_of_files = image.list_pictures(image_dir)
    print("Count of image files read is ",  len(list_of_files))

    # Iterate over each images and read their sizes
    list_images = []
    for img_name in list_of_files: #[0:10] img_name = list_of_files[0]
        img_path = img_name
        img = np.NaN
        #img = image.load_img(img_path, target_size=(W, H))
        if W > 0 and H > 0:
            img = image.load_img(img_path, target_size=(W, H))
        else:
            img = image.load_img(img_path)

        x = image.img_to_array(img)
        x /= image_array_divided_by
        list_images.append(x)
        del(img, x)

    # Stack to have one complete list as required by NN
    train = np.stack(list_images)

    # Cleaning
    del(list_images)

    # Return all required
    return({"train" : train, "list_of_files" : list_of_files})
    # end of load_images

# Description: Return 3D shape based on K Channel. Need to be made 4d before Analysis
def get_input_shape(count, img_rows, img_cols):
    if K.image_data_format() == 'channels_first':
        input_shape = (count, img_rows, img_cols)
    else: # "Count - Height-Width-Depth"
        input_shape = (img_rows, img_cols, count)

    return(input_shape)

# Description: It displays many images in one row
def imshow_all(*images, titles=None):
    images = [img_as_float(img) for img in images]

    if titles is None:
        titles = [''] * len(images)
    vmin = min(map(np.min, images))
    vmax = max(map(np.max, images))
    ncols = len(images)
    height = 5
    width = height * len(images)
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(width, height))
    for ax, img, label in zip(axes.ravel(), images, titles):
        ax.imshow(img, vmin=vmin, vmax=vmax)
        ax.set_title(label)

    return
    # end of imshow_all

# Description: It displays many images in many rows and column
# cmap='gray'; row_plot = 1; img= [img, camera]
def show_image(img, cmap='gray', row_plot = 1):
    if not isinstance(img, list):
        img = [img]

    if len(img) == 1:
        plt.imshow(img[0], cmap=cmap) # cm.gray
    else:
        # Get column count
        col_plot = len(img) // row_plot
        fig, axes = plt.subplots(row_plot, col_plot)
        for count in range(len(img)): # count= 1
            axes[count].imshow(img[count], cmap=cmap)
    plt.show()
    return
# end of show_image

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

#%% cv2 related
def cv2_show_fullscr(window_name, list_my_image_color):
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#    if len(list_my_image_color) > 1:
#    list_my_image_color = np.hstack(list_my_image_color)

    cv2.imshow(window_name, list_my_image_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# end of cv2_show_fullscr

# Take help from http://answers.opencv.org/question/99282/does-findcontours-create-duplicates/
# Description: It drop duplicate with area overlap of 'epsilon' overlap.
# There might be small contours - suppress all of them if their area is less than 'area_suppress'
def cv2_get_unique_contours(my_image_color, contours, epsilon =0.05, area_suppress = 100):
    # create dummy image with all 0
    mask = np.zeros(my_image_color.shape,np.uint8)

    # The multiplication may become big and hence scaling to 0-1
    my_image_color_temp = my_image_color.copy().astype(np.float)/my_image_color.max().astype(np.float)

    #get unique contours
    contours_new = []
    for contour in contours: # cv2.boundingRect(contours[7])
        x,y,w,h = cv2.boundingRect(contour)
        if area_suppress < w*h and np.sum(mask[y:y+h,x:x+w] * my_image_color_temp[y:y+h,x:x+w]) < epsilon * np.sum(my_image_color_temp[y:y+h,x:x+w]):
           contours_new.append(contour)
           mask[y:y+h,x:x+w] = 1

    del(my_image_color_temp, mask)

    return(contours_new)
#end of cv2_get_unique_contours

