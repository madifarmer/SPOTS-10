import cv2
import os


def extract_patch(image, center, patch_size, scaling_factor):
    if center is None:
        return None

    x, y = center
    x_original, y_original = int(x / scaling_factor), int(y / scaling_factor)

    half_size = int(patch_size / (2 * scaling_factor))

    # Ensure the patch coordinates are within the image boundaries
    if (x_original - half_size < 0 or x_original + half_size >= image.shape[1] or
            y_original - half_size < 0 or y_original + half_size >= image.shape[0]):
        return None

    patch = image[y_original - half_size:y_original + half_size, x_original - half_size:x_original + half_size]
    return patch


def click_callback(event, x, y, flags, param):
    global click_position, mouse_down
    if event == cv2.EVENT_LBUTTONDOWN:
        click_position = (x, y)
        mouse_down = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_down = False


def main():
    spotted_animal = "tapir"
    split = "train"

    input_folder = "../spotted_animals/" + split + "/" + spotted_animal
    output_folder = "../spotted_animals/" + split + "/" + spotted_animal + "_90_click"
    patch_size = 90

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_file in image_files:
        original_image = cv2.imread(os.path.join(input_folder, image_file))

        if original_image is None:
            print(f"Error loading image: {image_file}")
            continue

        global click_position, mouse_down
        click_position = None
        mouse_down = False

        # Resize the image to 800x600 if it's larger
        #if original_image.shape[0] > 600 or original_image.shape[1] > 800:
        original_image = cv2.resize(original_image, (800, 600))
        # Calculate the scaling factor based on the original and displayed image sizes
        scaling_factor = max(original_image.shape[0] / 600, original_image.shape[1] / 800)


        cv2.namedWindow("Image")
        cv2.setMouseCallback("Image", click_callback)

        patch_id = 0
        while True:
            cv2.imshow("Image", original_image)

            key = cv2.waitKey(1)
            if key == 27:  # Press 'Esc' to exit
                cv2.destroyAllWindows()
                exit()
            elif key == 13:  # Press 'Enter' to move to the next image
                cv2.destroyAllWindows()
                break

            if mouse_down and click_position is not None:
                patch = extract_patch(original_image, click_position, patch_size, scaling_factor)

                if patch is not None and patch.size != 0:  # Check if the patch is valid
                    output_path = os.path.join(output_folder, f"{image_file}_patch_{patch_id}.png")
                    cv2.imwrite(output_path, patch)
                    print(f"Patch saved: {output_path}")
                    patch_id += 1

                # Reset click_position to avoid saving multiple patches for the same click event
                click_position = None

if __name__ == "__main__":
    main()
