#     Evaluation Metrics
from itertools import cycle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix
from skimage.measure import label, regionprops, regionprops_table



def roc_auc_plot(y_true, y_pred,
                 n_classes=2, lw=2, data_title='Unspecified',
                 save_fig=False, fig_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Models/Figures'):
    """
    plot ROC and compute auc using sklearn. Taken from https://github.com/Tony607/ROC-Keras/blob/master/ROC-Keras.ipynb.
    :param y_true: ground truth
    :param y_pred: prediction
    :param n_classes: number of classes
    :param lw: line width
    :param data_title: title of the figure
    :param save_fig: True/False; If True then saves the figure to fig_dir
    :param fig_dir: directory of the saved figure
    :return:
    """

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(1)
    i = 1
    plt.plot(fpr[i], tpr[i], color='cornflowerblue', lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC--{data_title}")
    plt.legend(loc="lower right")

    # if save_fig:
    #     os.chdir(fig_dir)
    #     plt.savefig(f"ROC--{data_title}")

    # plt.show()

    return plt.figure(1)


# def dr_friendly_measures(y_true, y_pred):
#     """
#     :param y_true: ground truth
#     :param y_pred: prediction
#     :return: specificity, sensitivity, ppv, npv
#     """
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#
#     print('')
#     print('Summary of true values')
#     true_values = pd.DataFrame(y_true).rename(columns={0: 'true'})
#     print(true_values.groupby('true')['true'].count())
#
#     print('')
#     print('Patients needing Feeding Tubes')
#     print(f"True positives: {tp}")
#     print(f"False negatives: {fn}")
#     print('')
#     print('Patients not needing Feeding Tubes')
#     print(f"False positives: {fp}")
#     print(f"True negatives: {tn}")
#     print('')
#
#     specificity = tn / (tn + fp)
#     sensitivity = tp / (tp + fn)
#     ppv = tp / (tp + fp)
#     npv = tn / (tn + fn)
#
#     name1 = ['specificity', 'sensitivity', 'ppv', 'npv']
#     name2 = [specificity, sensitivity, ppv, npv]
#
#     for i in range(len(name1)):
#         print(f"{name1[i]}: {np.round(name2[i], 3)}")
#
#     return specificity, sensitivity, ppv, npv
#
#
# def fbeta(y_true, y_pred, beta=1):
#     return fbeta_score(y_true=y_true, y_pred=y_pred, beta=beta)

def cbct_ctsim_dose_image_review(cbct, ctsim, dose,
                                 save_fig_name, view=False,
                                 fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/saved_images/'):
    cbct_idx_list = [1, 4, 7, 10, 13, 16]
    ct_idx_list = [2, 5, 8, 11, 14, 17]
    # dose_dx_list = [3, 6, 9, 12, 15]

    fig = plt.figure(figsize=(20, 20))
    columns = 3
    rows = 6
    fig.subplots_adjust(hspace=0, wspace=-0.7)

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        if i in cbct_idx_list:
            # print(f"{(i)*5+10}")
            img = cbct[..., i * 4 + 0]
            img_max = cbct.max()
            if i == 1:
                plt.title(f"cbct")
        elif i in ct_idx_list:
            # print(f"{(i-1)*5+10}")
            img = ctsim[..., (i - 1) * 4 + 0]
            img_max = ctsim.max()
            if i == 2:
                plt.title(f"ct sim")
        else:
            # print(f"{(i-2)*5+10}")
            img = dose[..., (i - 2) * 4 + 0]
            img_max = dose.max()
            if i == 3:
                plt.title(f"dose")

        plt.imshow(img, vmax=img_max)
        plt.colorbar()
        plt.grid()
    plt.suptitle('CBCT, CTsim, and Dose Summary', fontsize=20)
    if view:
        plt.show()
    fig.savefig(os.path.join(fig_storage_dir, save_fig_name + f"_{np.random.randint(0, 10000)}"))
    plt.close(fig)

def three_image_review(cbct, ctsim, dose,
                                 save_fig_name, view=False,
                                 fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/saved_images/'):
    cbct_idx_list = [1, 4, 7, 10, 13, 16]
    ct_idx_list = [2, 5, 8, 11, 14, 17]
    # dose_dx_list = [3, 6, 9, 12, 15]

    fig = plt.figure(figsize=(20, 20))
    columns = 3
    rows = 6
    fig.subplots_adjust(hspace=0, wspace=-0.7)

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        if i in cbct_idx_list:
            # print(f"{(i)*5+10}")
            img = cbct[..., i * 4 + 0]
            img_max = cbct.max()
            if i == 1:
                plt.title(f"cbct")
        elif i in ct_idx_list:
            # print(f"{(i-1)*5+10}")
            img = ctsim[..., (i - 1) * 4 + 0]
            img_max = ctsim.max()
            if i == 2:
                plt.title(f"ct sim")
        else:
            # print(f"{(i-2)*5+10}")
            img = dose[..., (i - 2) * 4 + 0]
            img_max = dose.max()
            if i == 3:
                plt.title(f"dose")

        plt.imshow(img, vmax=img_max)
        plt.colorbar()
        plt.grid()
    plt.suptitle('CBCT, CTsim, and Dose Summary', fontsize=20)
    if view:
        plt.show()
    fig.savefig(os.path.join(fig_storage_dir, save_fig_name + f"_{np.random.randint(0, 10000)}"))
    plt.close(fig)


def ct_mask_image_review(ct, mask,
                         save_fig_name, view=False,
                         fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/saved_images/'):
    ct_index = [1]
    mask_index = [2]

    fig = plt.figure(figsize=(20, 20))
    columns = 1
    rows = 2
    fig.subplots_adjust(hspace=0, wspace=0)

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        if i in ct_index:
            # print(f"{(i)*5+10}")
            img = ct[...]
            img_max = ct.max()
            if i == 1:
                plt.title(f"ct; min {np.min(ct)}, max {np.max(ct)}")
        elif i in mask_index:
            img = mask[...]
            img_max = mask.max()
            if i == 2:
                plt.title(f"mask, min {np.min(mask)}, max {np.max(mask)}")

        plt.imshow(img, vmax=img_max)
        plt.colorbar()
        plt.grid()
    plt.suptitle('CT and Maks', fontsize=20)
    if view:
        plt.show()
    fig.savefig(os.path.join(fig_storage_dir, save_fig_name))
    plt.close(fig)

def multi_reg_image_review(img1, img2, img3, mrn, view=False, gaps=7, initial=5, image_rows=4,
                           fig_storage_dir=r'C:\Users\s185479\Desktop\check_image'):

    if not os.path.exists(fig_storage_dir):
        os.mkdir(fig_storage_dir)
    else:
        pass

    gaps = gaps
    initial = initial

    fig = plt.figure(figsize=(40, 40))
    columns = 3
    rows = image_rows
    fig.subplots_adjust(hspace=0.1, wspace=0.3)

    img1_idx_list = [1 + j * 3 for j in range(rows)]
    img2_idx_list = [2 + j * 3 for j in range(rows)]
    img3_idx_list = [3 + j * 3 for j in range(rows)]

    # setting up technique to alternate pet and ct images; ctsim and dose images

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)

        # adding ct image (switches with pet)
        if i in img1_idx_list:
            # print(f"{(i)*5+10}")
            img = img1[..., i * gaps + initial]
            img_max = img1.max()
            if i == 1:
                plt.title(f"img1")

        # adding ctsim (switches with dose)
        elif i in img2_idx_list:
            # print(f"{(i-1)*5+10}")
            img = img2[..., (i - 1) * gaps + initial]
            img_max = img2.max()
            if i == 2:
                plt.title(f"img2")

        elif i in img3_idx_list:
            # print(f"{(i-2)*5+10}")
            img = img3[..., (i - 2) * gaps + initial]
            img_max = img3.max()
            if i == 3:
                plt.title(f"img3")
        else:
            pass

        plt.imshow(img, vmax=img_max)
        plt.colorbar()
        plt.grid()
    plt.suptitle('Multi-image comparison', fontsize=20)
    if view:
        plt.show()
    fig.savefig(os.path.join(fig_storage_dir, f"{mrn}.png"))
    plt.close(fig)


def ct_mask_pred_image_review(ct, mask, pred,
                              save_fig_name, view=False,
                              fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/saved_images/'):
    ct_index = [1]
    mask_index = [2]
    pred_index = [3]

    fig = plt.figure(figsize=(20, 20))
    columns = 1
    rows = 3
    fig.subplots_adjust(hspace=0, wspace=0)

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        if i in ct_index:
            # print(f"{(i)*5+10}")
            img = ct[...]
            img_max = ct.max()
            img_min = ct.min()
            cmap = 'gray'
            if i == 1:
                plt.title(f"ct; min {np.min(ct)}, max {np.max(ct)}")
        elif i in mask_index:
            img = mask[...]
            # assert mask.min() >= 0.0, f"check label min; {mask.min()}"
            # assert mask.max() <= 1.0, f"check label max; {mask.max()}"
            img_max = 1
            img_min = 0
            cmap = 'viridis'
            if i == 2:
                plt.title(f"mask, min {np.min(mask)}, max {np.max(mask)}")
        elif i in pred_index:
            img = pred[...]
            img_max = 1
            img_min = 0
            cmap = 'viridis'
            if i == 3:
                plt.title(f"pred, min {np.min(pred)}, max {np.max(pred)}")

        plt.imshow(img, vmin=img_min, vmax=img_max, cmap=cmap)
        plt.colorbar()
        plt.grid()
    plt.suptitle('CT, Mask, Pred', fontsize=20)
    if view:
        plt.show()
    fig.savefig(os.path.join(fig_storage_dir, save_fig_name))
    plt.close(fig)


def ct_mask_pred_25d_image_review(ct, mask, pred, prior_pred,
                              save_fig_name, view=False,
                              fig_storage_dir=r'/data/maia/mdohop/Holman_Pathway/Feeding_Tube/Pytorch/Data/saved_images/'):
    ct_index = [1]
    mask_index = [2]
    pred_index = [3]
    prior_pred_index = [4]

    fig = plt.figure(figsize=(20, 20))
    columns = 1
    rows = 4
    fig.subplots_adjust(hspace=0, wspace=0)

    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        if i in ct_index:
            # print(f"{(i)*5+10}")
            img = ct[...]
            img_max = ct.max()
            img_min = ct.min()
            cmap = 'gray'
            if i == 1:
                plt.title(f"ct; min {np.min(ct)}, max {np.max(ct)}")
        elif i in mask_index:
            img = mask[...]
            # assert mask.min() >= 0.0, f"check label min; {mask.min()}"
            # assert mask.max() <= 1.0, f"check label max; {mask.max()}"
            img_max = 1
            img_min = 0
            cmap = 'viridis'
            if i == 2:
                plt.title(f"mask, min {np.min(mask)}, max {np.max(mask)}")
        elif i in pred_index:
            img = pred[...]
            img_max = 1
            img_min = 0
            cmap = 'viridis'
            if i == 3:
                plt.title(f"pred, min {np.min(pred)}, max {np.max(pred)}")
        elif i in prior_pred_index:
            img = prior_pred[...]
            img_max = 1
            img_min = 0
            cmap = 'viridis'
            if i == 4:
                plt.title(f"prior pred, min {np.min(prior_pred)}, max {np.max(prior_pred)}")

        plt.imshow(img, vmin=img_min, vmax=img_max, cmap=cmap)
        plt.colorbar()
        plt.grid()
    plt.suptitle('CT, Mask, Pred, Prior Pred', fontsize=20)
    if view:
        plt.show()
    fig.savefig(os.path.join(fig_storage_dir, save_fig_name))
    plt.close(fig)


def box_whisker_list_input(data, labels, fig_title):
    assert isinstance(data, list), 'data needs to be in list'
    assert isinstance(labels, list), 'labels need to be list'
    assert len(data) == len(labels), f"need labels for data; data len: {len(data)}, label len: {len(labels)}"

    fig7, ax7 = plt.subplots()
    ax7.set_title(fig_title)
    ax7.boxplot(data)
    initial_x_axis = [x + 1 for x in range(len(labels))]
    new_labels = [f"{x}\n{len(data[i])}" for i, x in enumerate(labels)]
    plt.xticks(initial_x_axis, new_labels)

    return fig7


def create_2d_bbox(mask, show_images=False):
    assert len(np.shape(mask)) == 2, f"check shape of mask: {np.shape(mask)}; should be 2d"

    label_img = label(mask)
    regions = regionprops(label_img)

    if show_images:
        plt.imshow(mask)
        plt.title('no box')

    fig, ax = plt.subplots()

    boxes = []
    store_dict = {f"region_{i}": None for i in range(len(regions))}

    for i, props in enumerate(regions):
        minr, minc, maxr, maxc = props.bbox
        # print(props.bbox)
        # print(np.shape(props.coords))
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        if show_images:
            ax.imshow(image)
            ax.plot(bx, by, '-r', linewidth=2.5)
            plt.title('box')

        boxes.append([minc, minr, maxc, maxr])
        store_dict[f"region_{i}"] = {'y': (minr, maxr), 'x': (minc, maxc)}

    return boxes

def create_coco_2d_bbox(mask, show_images=False):
    # The COCO bounding box format is [top left x position, top left y position, width, height].
    assert len(np.shape(mask)) == 2, f"check shape of mask: {np.shape(mask)}; should be 2d"

    label_img = label(mask)
    regions = regionprops(label_img)

    if show_images:
        plt.imshow(mask)
        plt.title('no box')

    fig, ax = plt.subplots()

    boxes = []
    store_dict = {f"region_{i}": None for i in range(len(regions))}

    for i, props in enumerate(regions):
        minr, minc, maxr, maxc = props.bbox
        # print(props.bbox)
        # print(np.shape(props.coords))
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        if show_images:
            ax.imshow(image)
            ax.plot(bx, by, '-r', linewidth=2.5)
            plt.title('box')

        boxes.append([minc, minr, maxc-minc, maxr-minr])
        store_dict[f"region_{i}"] = {'y': (minr, maxr), 'x': (minc, maxc)}

    return boxes