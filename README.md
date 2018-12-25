## Object detection using Faster-RCNN in Docker
A dockerized implementation of Faster-RCNN object detection running on gunicorn.

### Prerequisite
- The tensorflow models zoo available here https://github.com/tensorflow/models. The downloaded path should be used instead of `/srv/downloads/ml-datasets/tf-models`.
- The already trained Faster-RCNN model named faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28
 can be downloaded from [here][1].

[1]: http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

### Build command
```
$ docker build -t faster-rcnn .
```

### Run command

Run with interactive shell access to container
```
$ docker run -it \
    -v /srv/downloads/ml-datasets/tf-models:/usr/src/app/tf-models:ro \
    -v /srv/downloads/ml-datasets/pretrained-models/faster_rcnn_resnet_v2:/usr/src/app/models:ro \
    faster-rcnn bash
```

Run to test the app in action. We needed to increase the memory limit using `-m 4g` for the object detection model to work.
```
$ docker run \
    -v /srv/downloads/ml-datasets/tf-models:/usr/src/app/tf-models:ro \
    -v /srv/downloads/ml-datasets/pretrained-models/faster_rcnn_resnet_v2:/usr/src/app/models:ro \
    -p 10080:10080 \
    -m 4g faster-rcnn
```