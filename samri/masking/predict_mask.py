
def predict_mask(in_file):
    import os
    from os import path
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import nibabel as nib
    from samri.masking import utils
    import numpy as np
    import cv2
    from tensorflow import keras
    import tensorflow.keras.backend as K

    def dice_coef(y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A)+sum(B))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)

        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(y_true, y_pred):
        return 1 - dice_coef(y_true, y_pred)

    prediction_shape = (64, 64)
    image = nib.load(in_file)
    in_file_data = image.get_data()
    in_file_data = np.moveaxis(in_file_data, 2, 0)

    ori_shape = in_file_data.shape
    delta_shape = tuple(np.subtract(prediction_shape, ori_shape[1:]))

    model_path = '/home/hendrik/src/mlebe/old_results/new_new_hope3/dice_600_2019-12-18/1_Step/unet_ep381_val_loss0.05.hdf5'

    model = keras.models.load_model(model_path, custom_objects={'dice_coef_loss': dice_coef_loss})

    in_file_data = utils.preprocess(in_file_data, prediction_shape, 'coronal', switched_axis= True)

    mask_pred = np.empty((ori_shape[0], prediction_shape[0], prediction_shape[1]))
    for slice in range(in_file_data.shape[0]):
        temp = np.expand_dims(in_file_data[slice], -1)  # expand dims for channel
        temp = np.expand_dims(temp, 0)  # expand dims for batch
        prediction = model.predict(temp, verbose = 0)
        prediction = np.squeeze(prediction)
        mask_pred[slice, ...] = np.where(prediction > 0.9, 1, 0)


    """
    Reconstruct to original image size 
    """
    resized = np.empty(ori_shape)
    for i, slice in enumerate(mask_pred):

        if delta_shape[0] < 0 and delta_shape[0] < 0:
            resized_mask = cv2.resize(slice, (ori_shape[2], ori_shape[1]))
            resized[i] = resized_mask
        elif delta_shape[0] < 0:
            temp = cv2.resize(slice, (prediction_shape[1]), ori_shape[1])
            resized_mask = temp[:, delta_shape[1]//2:ori_shape[2] + delta_shape[1]//2]
            resized[i] = resized_mask
        elif delta_shape[1] < 0:
            temp = cv2.resize(slice, (ori_shape[2], prediction_shape[0]))
            resized_mask = temp[delta_shape[0]//2:ori_shape[1] + delta_shape[0]//2, :]
            resized[i] = resized_mask
        elif delta_shape[0] < 0 and delta_shape[1] < 0:
            resized_mask = slice[delta_shape[0]//2:ori_shape[1] + delta_shape[0]//2, delta_shape[1]//2:ori_shape[2] + delta_shape[1]//2]
            resized[i] = resized_mask


    resized = np.moveaxis(resized, 0, 2)

    original_image = image.get_data()

    masked_image = np.multiply(original_image, resized)

    nii_path = 'masked_image.nii.gz'
    nii_path = path.abspath(path.expanduser(nii_path))
    masked_image = nib.Nifti1Image(masked_image, image.affine, image.header)
    nib.save(masked_image, nii_path)


    return nii_path




