import gradio as gr
import cv2
import numpy as np

#* 实现图像变换，缩放  旋转  平移  水平旋转



# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
    #* 矩阵变换 -- np.vstack 函数将仿射矩阵的底部追加 [0, 0, 1]，使其成为3x3矩阵
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])







# Function to apply transformations based on user inputs
    # input：上传图像，缩放比例，旋转角度，xy轴平移量，是否水平旋转
    '''
    图像转numpy
    pad  -- 防止变换时超出边界 -- pad白色部分
    变换（自己写）
    '''

def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)
    transformed_image = np.array(image)
    '''
    pad_size 是图像的最小尺寸的一半，用于给图像边界添加填充。
    image_new 是新建的图像，增加了白色边框，防止旋转或缩放时图像超出边界。
    原始图像放置在新的中心位置，周围填充白色。
    '''

    # todo : FILL: Apply Composition Transform
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
    # Get image dimensions
    h, w = image.shape[:2]

    # Calculate the center of the image
    center_x, center_y = w // 2, h // 2

    # Create the scale matrix
    scale_matrix = np.array([[scale, 0, 0], [0, scale, 0]])

    # Create the rotation matrix (around the center)
    theta = np.radians(rotation)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0]])

    # Create the translation matrix
    translation_matrix = np.array([[1, 0, translation_x], [0, 1, translation_y]])

    # Convert to 3x3 matrix by appending [0, 0, 1]
    scale_matrix_3x3 = to_3x3(scale_matrix)
    rotation_matrix_3x3 = to_3x3(rotation_matrix)
    translation_matrix_3x3 = to_3x3(translation_matrix)

    # Combine transformations: T = Translation * Rotation * Scale
    transform_matrix = translation_matrix_3x3 @ rotation_matrix_3x3 @ scale_matrix_3x3

    # Translate the image center to origin for rotation/scale
    transform_to_origin = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]])
    transform_back = np.array([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]])

    # Final transformation matrix
    final_transform = transform_back @ transform_matrix @ transform_to_origin

    # Apply the affine transformation using OpenCV
    transformed_image = cv2.warpAffine(image, final_transform[:2], (w, h), borderValue=(255, 255, 255))

    # Apply horizontal flip if required
    if flip_horizontal:
        transformed_image = cv2.flip(transformed_image, 1)

    return transformed_image











# Gradio Interface
#* gradio页面设计
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()
