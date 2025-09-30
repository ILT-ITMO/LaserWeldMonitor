# # WeldGUI/app.py
# from welding.api import app 

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=7860)
# WeldGUI/app.py
from welding.api import demo  # import the Gradio Blocks object

if __name__ == "__main__":
    # Launch Gradio interface
    demo.launch(server_name="0.0.0.0", server_port=7860)