import hopenet, utils
import torch
import torchvision
from torchvision import transforms
import dlib, cv2
from PIL import Image
import torch.nn.functional as F


model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
state_dict = torch.load('hopenet_robust_alpha1.pkl')
model.load_state_dict(state_dict)

transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])

model.cuda(1)
model.eval()
idx_tensor = [idx for idx in xrange(66)]
idx_tensor = torch.FloatTensor(idx_tensor).cuda(1)

cnn_face_detector = dlib.cnn_face_detection_model_v1('../../mmod_human_face_detector.dat')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    dets = cnn_face_detector(cv2_frame, 1)

    try:
        bb = max(dets, key = lambda face: face.rect.width()*face.rect.height())
        bb = bb.rect
    except:
        bb = dlib.rectangle(0,0,frame.shape[1],frame.shape[0])

    x_min = bb.left()
    y_min = bb.top()
    x_max = bb.right()
    y_max = bb.bottom()

    bbox_width = abs(x_max-x_min)
    bbox_height = abs(y_max-y_min)
    x_min -= 2 * bbox_width / 4
    x_max += 2 * bbox_width / 4
    y_min -= 3 * bbox_height / 4
    y_max += bbox_height / 4
    x_min = max(x_min, 0);
    y_min = max(y_min, 0)
    x_max = min(frame.shape[1], x_max);
    y_max = min(frame.shape[0], y_max)
    # Crop image
    img = cv2_frame[y_min:y_max, x_min:x_max]
    img = Image.fromarray(img)

    # Transform
    img = transformations(img)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
    img = img.cuda(1)

    # predict
    yaw, pitch, roll = model(img)
    yaw_predicted = F.softmax(yaw, dim=1)
    pitch_predicted = F.softmax(pitch, dim=1)
    roll_predicted = F.softmax(roll, dim=1)
    # Get continuous predictions in degrees.
    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
    utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx=(x_min + x_max) / 2,
                    tdy=(y_min + y_max) / 2, size=bbox_height / 2)

    cv2.imshow('camera', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
