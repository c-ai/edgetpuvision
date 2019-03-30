# edgetpuvision
Google cloud edge TPU edgetpu default package.

| What's up?    | Where to go |
|:--------------|:------------|
| Got a Coral dev board. What do I do? |  official [Getting Started Guide][1]. |
| Can't get something working. | The [issue tracker][0] is where we discuss problems. |
| Found something wrong in Google's docs. | Stick it in the [issue tracker][0] so we can feedback to Google. |
| Created something would like to share. | Stick it in the [Pull Request][1] so we can feedback to Google. |

# Overview

Officially, Google Coral is two devices:
```
Dev Board
A single-board computer with a removable system-on-module (SOM) featuring the Edge TPU.
*    Supported OS: Mendel Linux (derivative of Debian)
*    Supported Framework: TensorFlow Lite
*    Languages: Python (C++ coming soon)
```
and
```
USB Accelerator
A USB accessory featuring the Edge TPU that brings ML inferencing to existing systems.
*    Supported OS: Debian Linux
*    Compatible with Raspberry Pi boards
*    Supported Framework: TensorFlow Lite
```
Source: https://coral.withgoogle.com/

These devices are brand new, and as of right now there is very little supporting documentation(not mention no "Mendel Linux" could be found out there on the INTERNET). This repo exists to help us build up the community and help each others out.



[0]: https://github.com/CharlesCCC/edgetpuvision/issues
[1]: https://github.com/CharlesCCC/edgetpuvision/pulls
