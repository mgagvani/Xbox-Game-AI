Xbox Game AI
==========



Based on [TensorKart](https://github.com/kevinhughes27/TensorKart)[^1], adapted to use [PYXInput](https://github.com/bayangan1991/PYXInput) for control of Xbox/PC games. Has a CV-based method for driving an autonomous car inside of

## Getting Started
1. I would recommend using an Anaconda environment for easier library management. Once you have a Conda environment, enter it and clone this repository. Then, run `conda install --file requirements.txt` to install the necessary libraries.
2. Install Intel OpenVINO, this is needed for the road segmentation AI model used by default.
3. Out of the box, running `play.py` will attempt to find an image of a road from *Forza Horizon 3* and [](./samples/forza_road3/00121.png) and autonomously drive the car.
4. So, turn on your game, such as *Forza Horizon*, preferably on a secondary monitor and get your car in a stable position. Then run `play.py` and you should start seeing the car move after a few seconds. 

## Dependencies
If you already have a Python installation you want to use, here are the major dependencies:

* Numpy
* Tensorflow 2.x
* Intel OpenVINO
* PYXInput
* MatPlotLib
* OpenCV

## Standard AI Mode - Collect, Train, Test
TensorKart, which this is based upon, was designed to collect screenshots along with matching gamepad data, and learn what images on the screen correspond with what controller inputs using a CNN(convolutional neural network).

*Note*: The following is somewhat paraphrased from the TensorKart README. 

* Recording Training Data

    1. Start your game. 
    2. Make sure you have an Xbox controller connected, either through Bluetooth or wired, to your PC.
    3. Run `python record.py`, the graph should change according to user input. 
    4. If the game window is not properly captured by the preview on the left, try changing `SRC_W`, `SRC_H`, `OFFSET_X`, and `OFFSET_Y` to match the postion your game is launching at. 
    5. Press record to start recording samples. Press the button again to stop recording. 

*Note*: For most games, not all inputs are used, or are very important to the actual functioning of the game. This can result in extremely sparse data which is harder for AI models to use. To remove some controller inputs, try replacing line 301 of `utils.py ` with `load_mini_sample` or another method which operates in the same way as any of the other `load_sample()` methods. 

* Preparing Training Data

  1. Run `python utils.py prepare samples/*` to build the input and output data sets for training. You might want to temporarily move "bad" datasets to another folder before doing this so they are not included.

* Training the Model

  1. Run python train.py with the correct number of controller outputs under OUT_SHAPE. The program should run on an NVIDIA GPU automatically to speed up training time. It will automatically save the best model (based on validation loss) to a .h5 file. 

* Testing

  1. By default, `play.py` has some very low-level Windows-specific optimizations built in. Comment these out if you are not using Windows. 
  2. Change line 242 to `actor.act(pic)`, this is the general purpose Actor which will work with a normal sample and a mini sample.
  3. You can temporarily override the AI by pressing down on the RS (Right Stick).
   
## Future Improvements:
- [ ] Add reinforcement learning model for automatically learning how to drive a virtual *Forza* car
- [ ] Add an estimated perspective transform to find lane lines in *Forza* or other racing games
- [ ] Convert Intel OpenVINO model to Tensorflow to remove a dependency
- [ ] Streamline CNN-based process for different samples
- [ ] Improve documentation

## Contributing 🙈🙈
Don't hesitate to **open a pull request or issue** with functionality you want to see added, or bugs you have found!

[^1]: TensorKart uses a Gym environment. If you already have an OpenAI Gym environment for your game, you might want to take a look at that. This is more useful for games with an Xbox controller input.