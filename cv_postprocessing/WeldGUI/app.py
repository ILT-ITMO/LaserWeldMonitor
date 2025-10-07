from welding.api import demo  # import the Gradio Blocks object
import socket

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]
if __name__ == "__main__":
    # Launch Gradio interface
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860)
    except Exception as e:
        print(f"Failed on port 7860: {e}")
        free_port = find_free_port()
        print(f"Trying free port: {free_port}")
        demo.launch(server_name="0.0.0.0", server_port=free_port)