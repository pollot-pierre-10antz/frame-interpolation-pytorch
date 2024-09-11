import os
import gradio as gr

from inference import inference, select_device, load_model


this_dir = os.path.dirname(__file__)
output_folder_path = os.path.join(this_dir, "outputs")

output_count = 1
continue_counting = True

device = select_device("auto")
model = None

with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            model_path = gr.FileExplorer("**/*.pt", root_dir=os.path.join(this_dir, "checkpoints"), file_count="single", label="Model path")    
            device_dropdown = gr.Dropdown(["auto", "gpu", "cpu"], value="auto", interactive=True, label="Device")
            with gr.Row():
                fp16 = gr.Checkbox(value=False, label="fp16")
                frames = gr.Number(18, precision=0, minimum=1, label="Number of frames to interpolate")
                fps = gr.Number(10, precision=0, minimum=1, label="FPS")
            with gr.Row():
                frame1 = gr.Image(
                    #value=os.path.join(this_dir, "photos/one.png"),
                    type="filepath", 
                    label="Frame 1",
                )
                frame2 = gr.Image(
                    #value=os.path.join(this_dir, "photos/two.png"),
                    type="filepath", 
                    label="Frame 2",
                )
            save_path = gr.Text(os.path.join(output_folder_path, f"{output_count}.mp4"), interactive=True, label="Save path")
        with gr.Column():
            video = gr.Video(label="Output video")
    with gr.Row():
        generate_button = gr.Button(value="Generate video", variant="primary")
        
    def model_loading_trigger(checkpoint_path: str | None, is_fp16: bool):
        if checkpoint_path is not None:
            global model, device
            model = load_model(checkpoint_path, device, is_fp16)
            print("Model loaded/re-loaded.")
        else: 
            model = None
            print("Model unloaded.")
        
    model_path.change(model_loading_trigger, inputs=[model_path, fp16])
    fp16.change(model_loading_trigger, inputs=[model_path, fp16])
    
    
    def device_loading_trigger(device_name: str, checkpoint_path: str, is_fp16: bool):
        global device
        force_settings = {}
        try:
            device = select_device(device_name)
        except RuntimeError:
            device = select_device("auto")
            print(f"{device_name} cannot be loaded. Switch to device {device}")
            force_settings["value"] = "auto"
        model_loading_trigger(checkpoint_path, is_fp16)
        return gr.update(**force_settings)
    
    device_dropdown.change(device_loading_trigger, inputs=[device_dropdown, model_path, fp16], outputs=device_dropdown)
    
    
    def trigger_inference(frame1, frame2, save_path, frames, fps, is_fp16):
        if model is None:
            print("The model must be loaded first to generate the video.")
            return [gr.update(), gr.update()]
        inference(model, device, frame1, frame2, save_path, frames, fps, is_fp16)
        if continue_counting:
            global output_count
            output_count += 1
            next_save_file_name = os.path.join(output_folder_path, f"{output_count}.mp4")
            video_update = gr.update(value=save_path)
            return [gr.update(value=next_save_file_name), video_update]
        return [gr.update(), video_update]
        
    generate_button.click(
        lambda: gr.update(interactive=False),
        outputs=generate_button,
    ).then(
        trigger_inference,
        inputs=[frame1, frame2, save_path, frames, fps, fp16],
        outputs=[save_path, video],
    ).then(
        lambda: gr.update(interactive=True),
        outputs=generate_button,
    )
    
    
    def stop_counting_trigger():
        continue_counting = False
    
    save_path.change(stop_counting_trigger)


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--port", "-p", type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--share", action="store_true")
    
    args = parser.parse_args()
    
    app.launch(
        debug=args.debug,
        server_port=args.port,
        share=args.share,
    )