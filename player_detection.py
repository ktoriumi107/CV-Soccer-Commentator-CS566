from inference import get_model

def run_on_image(img_path):

    # load the model
    model = get_model(
        model_id="Soccer Commentator/soccer player/1",
        api_key="5UnSJbTztgfXiaW9zfbv"
    )

    # run the model
    results = model.infer(img_path)

    print(results)

def run_on_video():
    pass

run_on_image("test_images/orthogonal_field_view.png")