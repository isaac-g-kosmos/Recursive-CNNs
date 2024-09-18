from matplotlib import pyplot as plt

import dataprocessor
import pandas as pd
import numpy as np

dataset_type = "document"
# data_dirs = [
#     r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\augmentations",
#     r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\kosmos-dataset",
#     r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\self-collected",
#     r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets\testDataset\smart-doc-train",
#
#
# ]
# dataset = dataprocessor.DatasetFactory.get_dataset(data_dirs, dataset_type, "test.csv")
#
#
# loader_dataset =dataprocessor.LoaderFactory.get_loader("hdd", dataset.myData,
#                                               transform=None,
#                                               cuda=False)
predictions = pd.read_csv(
    r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\experiment-7-doc\results.csv")

corner_to_unpack = ["top_left",
                    "top_right",
                    "bottom_right",
                    "bottom_left"]
# %%
for column in corner_to_unpack:
    predictions[column + "_pred_x"] = predictions[column].apply(lambda x: eval(x)[0][0])
    predictions[column + "_pred_y"] = predictions[column].apply(lambda x: eval(x)[0][1])
    predictions[column + "_x"] = predictions[column].apply(lambda x: eval(x)[1][0])
    predictions[column + "_y"] = predictions[column].apply(lambda x: eval(x)[1][1])
    predictions[column + "_y_lower_bound"] = predictions[column].apply(lambda x: eval(x)[2])
    predictions[column + "_y_upper_bound"] = predictions[column].apply(lambda x: eval(x)[3])
    predictions[column + "_x_lower_bound"] = predictions[column].apply(lambda x: eval(x)[4])
    predictions[column + "_x_upper_bound"] = predictions[column].apply(lambda x: eval(x)[5])


# %%
def calculate_area(lower_bound_y, upper_bound_y, lower_bound_x, upper_bound_x):
    if float(lower_bound_y) > float(upper_bound_y):
        return 0
    if float(lower_bound_x) > float(upper_bound_x):
        return 0
    return (float(upper_bound_y) - float(lower_bound_y)) * (float(upper_bound_x) - float(lower_bound_x))


def empty(x):
    return x


predictions["top_left_area"] = 0
predictions["top_right_area"] = 0
predictions["bottom_right_area"] = 0
predictions["bottom_left_area"] = 0

for idx, row in predictions.iterrows():

    for column in corner_to_unpack:
        # predictions[column + "_area"] = predictions.apply(lambda x: print(x))
        predictions.loc[idx, column + "_area"] = calculate_area(row[column + "_y_lower_bound"],
                                                                row[column + "_y_upper_bound"],
                                                                row[column + "_x_lower_bound"],
                                                                row[column + "_x_upper_bound"])
        # predictions[column + "_area"] = calculate_area(predictions[column + "y_lower_bound"],
        #                                                predictions[column + "y_upper_bound"],
        #                                                predictions[column + "x_lower_bound"],
        #                                                predictions[column + "x_upper_bound"])
# %%
change_dictionary = {
    "top_left": "contains_" + "tl",
    "top_right": "contains_" + "tr",
    "bottom_right": "contains_" + "br",
    "bottom_left": "contains_" + "bl",
}
from matplotlib import pyplot as plt

for column in corner_to_unpack:
    contains_colums = change_dictionary[column]
    predictions_finds = predictions[predictions[contains_colums] == 1]
    predictions_unfinds = predictions[predictions[contains_colums] == 0]
    # column_area=predictions_finds[column+"_area"]
    fig, ax = plt.subplots(2, 1)
    ax[0].hist(predictions_finds[column + "_area"])
    ax[0].set_title(column + "found")
    ax[1].hist(predictions_unfinds[column + "_area"])
    ax[1].set_title(column + "found")
    print("unfound " + column + "_area")
    print(np.percentile(predictions_unfinds[column + "_area"], [10, 50, 80, 95, 100]))
    print("found " + column + "_area")
    print(np.percentile(predictions_finds[column + "_area"], [3, 5, 10, 20, 30, 40, 50, 90, 100]))
    print("//////////////")
    plt.show()


# %%


# %%
def euclidean_distance_np(x1, y1, x2, y2):
    # Convert to NumPy arrays if they aren't already
    x1, y1, x2, y2 = np.array(x1), np.array(y1), np.array(x2), np.array(y2)
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


predictions["predicted_width"] = 0
predictions["predicted_height"] = 0

predicted_top_width = euclidean_distance_np(
    predictions["top_left_pred_x"],
    predictions["top_left_pred_y"],
    predictions["top_right_pred_x"],
    predictions["top_right_pred_y"],
)
predicted_bottom_width = euclidean_distance_np(
    predictions["bottom_left_pred_x"],
    predictions["bottom_left_pred_y"],
    predictions["bottom_right_pred_x"],
    predictions["bottom_right_pred_y"],
)
predicted_width = (predicted_top_width + predicted_bottom_width) / 2

predicted_left_height = euclidean_distance_np(
    predictions["top_left_pred_x"],
    predictions["top_left_pred_y"],
    predictions["bottom_left_pred_x"],
    predictions["bottom_left_pred_y"]
)

predicted_right_height = euclidean_distance_np(
    predictions["top_right_pred_x"],
    predictions["top_right_pred_y"],
    predictions["bottom_right_pred_x"],
    predictions["bottom_right_pred_y"]
)
predicted_height = (predicted_left_height + predicted_right_height) / 2

predictions["predicted_width"] = predicted_width
predictions["predicted_height"] = predicted_height
# %%
for column in corner_to_unpack:
    predictions["valid_region_" + column] = (predictions[column + "_area"] > .05)


# %%
def cordinate_within_intervals(cordinate, x_interval, y_interval) -> int:
    is_within_x = (x_interval[0] <= cordinate[0] <= x_interval[1])
    is_within_y = (y_interval[0] <= cordinate[1] <= y_interval[1])

    return int(is_within_x and is_within_y)


# %%
scale = .25

accs = []
acc_3s = []
acc_4s = []
scales = []


def calculate_new_metric(df: pd.DataFrame, scale):
    df = df.copy()
    for column in corner_to_unpack:
        df["new_" + column + "_y_lower_bound"] = predictions[column + "_y_lower_bound"]
        df["new_" + column + "_y_upper_bound"] = predictions[column + "_y_upper_bound"]
        df["new_" + column + "_x_lower_bound"] = predictions[column + "_x_lower_bound"]
        df["new_" + column + "_x_upper_bound"] = predictions[column + "_x_upper_bound"]

        df["new_" + column + "_y_lower_bound"] = predictions[column + "_pred_y"] - scale * predictions[
            "predicted_height"]
        df["new_" + column + "_y_upper_bound"] = predictions[column + "_pred_y"] + scale * predictions[
            "predicted_width"]
        df["new_" + column + "_x_lower_bound"] = predictions[column + "_pred_x"] - scale * predictions[
            "predicted_height"]
        df["new_" + column + "_x_upper_bound"] = predictions[column + "_pred_x"] + scale * predictions[
            "predicted_width"]

    for column in corner_to_unpack:
        new_name = change_dictionary[column]
        # predictions["new_contains_"+new_name]=0
        for idx, row in df.iterrows():
            corner_result = cordinate_within_intervals((row[column + "_x"], row[column + "_y"]),
                                                       (row["new_" + column + "_x_lower_bound"],
                                                        row["new_" + column + "_x_upper_bound"]),
                                                       (row["new_" + column + "_y_lower_bound"],
                                                        row["new_" + column + "_y_upper_bound"]))
            df.loc[idx, "new_" + new_name] = corner_result

    df["new_total"] = np.array(df[["new_contains_tl", "new_contains_tr", "new_contains_br", "new_contains_bl", ]]).sum(
        axis=1)
    return df


for increase in range(11):
    scale = increase * .001 + scale
    scales.append(scale)

    new_preds = calculate_new_metric(predictions, scale)

    acc = np.sum(new_preds["new_total"]) / 4 / len(new_preds)
    acc_3 = np.sum((new_preds["new_total"]) == 4) / len(new_preds)
    acc_4 = np.sum((new_preds["new_total"]) >= 3) / len(new_preds)

    accs.append(acc)
    acc_3s.append(acc_3)
    acc_4s.append(acc_4)

    print("Corner accuracy")
    print(np.round(acc, 3))
    print("4 Corner accuracy")
    print(np.round(acc_3, 3))
    print("3 Corner accuracy")
    print(np.round(acc_4, 3))
# %%
result = pd.DataFrame({"scale": scales,
                       "accuracy": accs,
                       "accuracy_3_corners": acc_4s,
                       "accuracy_4_corners": acc_3s,

                       })

# %%
plt.plot(scales, accs, label="Corner accuracy")
plt.plot(scales, acc_4s, label="3 Corner accuracy")
plt.plot(scales, acc_3s, label="4 Corner accuracy")

plt.title('Accuracy vs scale')
plt.xlabel('Scale')
plt.ylabel('Accuracy')

# Display legend
plt.legend()

# Show the plot
plt.show()
# %%
new_cornes = calculate_new_metric(predictions, .25)
# %%

missing_cornes = new_cornes[new_cornes["new_total"] < 4]
all_corners = new_cornes[new_cornes["new_total"] == 4]
# %%
from PIL import Image
from PIL import ImageDraw



def draw_multiple_bounding_boxes(image, bounding_boxes):
    """
    Draws multiple bounding boxes on an image.

    :param image_path: Path to the input image.
    :param bounding_boxes: List of bounding boxes, each defined as [lower_bound_y, upper_bound_y, lower_bound_x, upper_bound_x].
    :param output_path: Path to save the output image.
    """
    # Load the image
    # image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Loop through each bounding box and draw it on the image
    for bounding_box in bounding_boxes:
        lower_bound_y, upper_bound_y, lower_bound_x, upper_bound_x = bounding_box
        rect_coords = [lower_bound_x*image.size[0], lower_bound_y*image.size[1], upper_bound_x*image.size[0], upper_bound_y*image.size[1]]
        try:
            draw.rectangle(rect_coords, outline="red", width=1)
        except:
            pass
    return image
def highlight_coordinates(image, coordinates,color,  radius=1):
    """
    Highlights coordinates on an image by drawing small green circles around them.

    :param image_path: Path to the input image.
    :param coordinates: List of coordinates, each defined as (x, y).
    :param output_path: Path to save the output image.
    :param radius: Radius of the circle used to highlight the coordinates. Default is 5.
    """
    # Load the image
    # image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Loop through each coordinate and draw a small circle around it
    for (x, y) in coordinates:
        left_up_point = ((x*image.size[0]- radius), (y*image.size[1]- radius))
        right_down_point = ((x*image.size[0]+ radius), (y*image.size[1]+ radius))
        draw.ellipse([left_up_point, right_down_point], outline=color, width=2, fill=color)
    return image

#%%
replace_path = r"C:\Users\isaac\PycharmProjects\document_localization\Recursive-CNNs\datasets"
sample = missing_cornes.sample(1)
# sample = all_corners.sample(1)
sample_path = sample["path"].iloc[0].replace(r'/home/ubuntu/document_localization/Recursive-CNNs/datasets',
                                             replace_path).replace("/","\\")

img=Image.open(sample_path)
bbs=[]
preds=[]
labels=[]
for column in corner_to_unpack:
   bbs.append(
       (sample["new_"+column+"_y_lower_bound"].iloc[0],
       sample["new_"+column+"_y_upper_bound"].iloc[0],
       sample["new_"+column+"_x_lower_bound"].iloc[0],
       sample["new_"+column+"_x_upper_bound"].iloc[0],)
              )
   labels.append(
       (sample[column+"_x"].iloc[0],
        sample[column+"_y"].iloc[0],
        )
   )
   preds.append(
       (sample[column+"_pred_x"].iloc[0],
        sample[column+"_pred_y"].iloc[0],
        )
   )
img=draw_multiple_bounding_boxes(img, bbs)
img = highlight_coordinates(img, preds, "green")
img = highlight_coordinates(img, labels, "blue")
plt.imshow(img)
plt.show()
