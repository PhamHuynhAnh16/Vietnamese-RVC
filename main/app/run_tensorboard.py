import logging
import webbrowser

from tensorboard import program


def launch_tensorboard_pipeline():
    logging.getLogger("root").setLevel(logging.ERROR)
    logging.getLogger("tensorboard").setLevel(logging.ERROR)

    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", "assets/logs", f"--port=6870"])
    url = tb.launch()

    print(f"Đường dẫn biểu đồ: {url}")
    
    webbrowser.open(url)


if __name__ == "__main__": launch_tensorboard_pipeline()