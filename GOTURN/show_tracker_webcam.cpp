#include <string>

#include <opencv2/opencv.hpp>

#include "network/regressor.h"
#include "tracker/tracker.h"

int main () {
  // Set up the neural network.
  const std::string model_file = "nets/tracker.prototxt";
  const std::string trained_file = "nets/models/pretrained_model/tracker.caffemodel";
  const int gpu_id = 0;
  const bool do_train = false;
  Regressor regressor(model_file, trained_file, gpu_id, do_train);

  // Set up the tracker
  const bool show_intermediate_output = false;
  Tracker tracker(show_intermediate_output);

  // Capture stream from webcam.
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    return 1;
  }

  // Create a window for display
  cv::namedWindow("TrackerStreamer", cv::WINDOW_AUTOSIZE);

  // Loop to capture frames from stream and track objects
  bool isFirstFrame = true;
  while (true) {
    cv::Mat frame;
    cap >> frame;

    if (isFirstFrame) {
      // Setup the first boudning box for initialization
      BoundingBox bbox;
      bbox.x1_ = 0;
      bbox.y1_ = 0;
      bbox.x2_ = 200;
      bbox.y2_ = 200;
      // Initialize the tracker.
      tracker.Init(frame, bbox, &regressor);
      // First frame is processed, set the flat to be false.
      isFirstFrame = false;
    } else {
      // Track and estimate the target's bounding box location in the current image.
      BoundingBox bbox_estimate_uncentered;
      tracker.Track(frame, &regressor, &bbox_estimate_uncentered);
      // Draw estimated bounding box of the target location (red).
      bbox_estimate_uncentered.Draw(255, 0, 0, &frame);
    }

    // Show the image with the estimated bounding boxes.
    cv::imshow("TrackerStreamer", frame);
    // Pause for 30 milliseconds.
    cv::waitKey(30);
  }

  return 0;
}
