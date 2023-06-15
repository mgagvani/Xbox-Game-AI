Xbox Game AI
============

**Our goal is to make this into a modular framework for making AIs with Xbox games that haven't been made compatible with Gym.**

Based on [TensorKart](https://github.com/kevinhughes27/TensorKart)[^1], adapted to use [PYXInput](https://github.com/bayangan1991/PYXInput) for control of Xbox/PC games. Has a CV-based method for driving an autonomous car inside of *Forza Horizon*.

## Getting Started

1. I would recommend using an Anaconda environment for easier library management. Once you have a Conda environment, enter it and clone this repository. Then, run `conda install --file requirements.txt` to install the necessary libraries.
2. Install Intel OpenVINO, this is needed for the road segmentation AI model used by default.
3. Out of the box, running `play.py` will attempt to find an image of a road from *Forza Horizon 3* and [](./samples/forza_road3/00121.png) and autonomously drive the car.
4. So, turn on your game, such as *Forza Horizon*, preferably on a secondary monitor and get your car in a stable position. Then run `play.py` and you should start seeing the car move after a few seconds.

*Note*: Make sure to use Python 3.6 if you are using `requirements.txt`.

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
  2. You can test your model by running `python utils.py plotpredictions y_data.npy model_weights_file.h5 x_data.npy categorical(True/False)`. This will run your model on every image in the dataset provided with the model provided and plot the ground truth vs. predicted outputs. 
* Testing

  1. By default, `play.py` has some very low-level Windows-specific optimizations built in. Comment these out if you are not using Windows.
  2. Change line 242 to `actor.act(pic)`, this is the general purpose Actor which will work with a normal sample and a mini sample.
  3. You can temporarily override the AI by pressing down on the RS (Right Stick).

### Categorical AI Mode

I have also experimented with another, much more complex video game - [NBA2K21](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D60q2Ct1ksBw&psig=AOvVaw0V2BQOkFlplYgN9YUmMxjT&ust=1637846756968000&source=images&cd=vfe&ved=0CAkQjhxqFwoTCOCDkuyMsfQCFQAAAAAdAAAAABAE). However, there are so many inputs, which are pressed so infrequently, that a standard model learns to do nothing (output values are always very close to 0). So, I manually classified each picture into 8 categories, for what the AI *should* do. These are defined in `play.py`, under "High-Level Control Methods". I then used transfer learning on EfficientNet to get a model specific to my game. Currently this part is a work in progress.

## What if I don't have the game on PC?

You can also use games that you have on an Xbox with this code, using a handy app called **Xbox Console Companion**. It comes bundled with most Windows installations, if it is not, download it from the Windows store.

### How to use it

1. Open Xbox Console Companion, and turn on your Xbox. Both the computer and XBox should be on the same network.
2. Go to "Connect to your Xbox One", and click on your Xbox. If it does not appear find the IP address of the Xbox and enter it.
3. You now have a live, 2-way stream from your computer to you Xbox! I have used this sucessfully to record lots of training data, it is quite reliable, even on a wireless network.

## Future Improvements:

- [ ] Add reinforcement learning model for automatically learning how to drive a virtual *Forza* car
- [ ] Add an estimated perspective transform to find lane lines in *Forza* or other racing games
- [ ] Convert Intel OpenVINO model to Tensorflow to remove a dependency
- [ ] Streamline CNN-based process for different samples
- [ ] Improve documentation

## Contributing

Don't hesitate to **open a pull request or issue** with functionality you want to see added, or bugs you have found!

[^1]: TensorKart uses a Gym environment. If you already have an OpenAI Gym environment for your game, you might want to take a look at that. This is more useful for games with an Xbox controller input.
