import numpy as np
import torch
import os
from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
from transform import Relabel, ToLabel, Colorize
import visdom
from torchvision.utils import save_image

from dataset import cityscapes
from dataset import VOCSegmentation

from lednet import Net


NUM_CHANNELS = 3
NUM_CLASSES = 2

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize((512,1024),Image.BILINEAR),
    ToTensor(),
    #Normalize([.485, .456, .406], [.229, .224, .225]),
])
target_transform_cityscapes = Compose([
    Resize((512,1024),Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   #ignore label to 19
])

input_transform = Compose([
    Resize((512,1024),Image.BILINEAR),
    ToTensor(),
])
target_transform = Compose([
    Resize((512,1024),Image.NEAREST),
    ToLabel(),
    Relabel(255, 0),   
])

class MyVOCTransform(object):
    def __init__(self):
        pass
    def __call__(self, input, target):
        input =  Resize((512, 1024),Image.BILINEAR)(input)
        input = ToTensor()(input)
        if target is not None:
            target = Resize((512, 1024),Image.NEAREST)(target)
            target =ToLabel()(target)
            target = Relabel(255, 0)(target)
        return input, target

cityscapes_trainIds2labelIds = Compose([
    Relabel(19, 255),  
    Relabel(18, 33),
    Relabel(17, 32),
    Relabel(16, 31),
    Relabel(15, 28),
    Relabel(14, 27),
    Relabel(13, 26),
    Relabel(12, 25),
    Relabel(11, 24),
    Relabel(10, 23),
    Relabel(9, 22),
    Relabel(8, 21),
    Relabel(7, 20),
    Relabel(6, 19),
    Relabel(5, 17),
    Relabel(4, 13),
    Relabel(3, 12),
    Relabel(2, 11),
    Relabel(1, 8),
    Relabel(0, 7),
    Relabel(255, 0),
    ToPILImage(),
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = Net(NUM_CLASSES)
  
    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()

    #model.load_state_dict(torch.load(args.state))
    #model.load_state_dict(torch.load(weightspath)) #not working if missing key

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict['state_dict'].items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
            print('Loaded param {}'.format(param))
        # print('state dict: {}  own_state {}'.format(state_dict['state_dict'].keys(), own_state.keys()))

        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print ("Model and weights LOADED successfully")

    model.eval()
    print('Model eval completed successfully')

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")

    voc_transform = MyVOCTransform()#512)
    dataset = VOCSegmentation(args.datadir, image_set=args.subset, transforms=voc_transform)

    loader = DataLoader(dataset,
    num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)


    # For visualizer:
    # must launch in other window "python3.6 -m visdom.server -port 8097"
    # and access localhost:8097 to see it
    if (args.visualize):
        vis = visdom.Visdom()
    
    # enumerator = (images, filename)
    # if os.path.isdir(os.path.join(args.datadir,args.subset, "labels")):
    #     enumerator = (images, labels, filename, filename)

    for step, (images, filename) in enumerate(loader):
        filtered_Filename = str(filename)
        filtered_Filename = filtered_Filename[2:len(filtered_Filename)-3]
        # print('Filtered FileName {}'.format(filtered_Filename))
        
        if (not args.cpu):
            images = images.cuda()
            #labels = labels.cuda()

        inputs = Variable(images)
        #targets = Variable(labels)
        with torch.no_grad():
            outputs = model(inputs)

        # print('output shape:', outputs.shape)
        # print('outputs:', outputs)
        # print('outputs[0].max(0) shape:', outputs[0].max(0))
        # print('outputs[0].max(0)[1] shape:', outputs[0].max(0)[1])
        label = outputs[0].max(0)[1].byte().cpu().data
        
        unique, count = torch.unique(label, return_counts=True)
        print('target unique {} and count {}'.format(unique, count))

        # print('label {}'.format(label))
        # print('label shape:', label.shape)
        label_color = Colorize()(label.unsqueeze(0))

        
        filtered_Filename += ".png"   
        filenameSave = "./save_color/" + args.subset + "/" + filtered_Filename

        os.makedirs(os.path.dirname(filenameSave), exist_ok=True)
        print('label shape: {} {}'.format(label.shape, type(label)))
        print('label_color shape: {} {}'.format(label_color.shape, type(label_color)))

        label = label*255.0
        unique, count = torch.unique(label, return_counts=True)
        print('label unique {} and count {}'.format(unique, count))

        # label_save = ToPILImage()(label_color)
        label_save = ToPILImage()(label.type(torch.uint8))              
        label_save.save(filenameSave)

        # save_image(label, filenameSave) 
        # save_image(label_color, filenameSave)
        
        if (args.visualize):
            vis.image(label_color.numpy())
        # print (step, filenameSave)
        

    print('model inference on all images saved successfully')
    

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')


    parser.add_argument('--loadDir',default="../save/logs/")
    parser.add_argument('--loadWeights', default="model_best.pth")
    parser.add_argument('--loadModel', default="lednet.py")
    parser.add_argument('--subset', default="val")  #can be val, test, train, demoSequence

    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--visualize', action='store_true')
    main(parser.parse_args())
