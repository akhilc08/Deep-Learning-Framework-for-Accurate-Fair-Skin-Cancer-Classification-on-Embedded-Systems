from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import os
from data import MyDataset
from torch.utils.data import DataLoader
 

def add_white_gaussian_noise(data, mean=0.0, std_dev=1.0):
    noise = np.random.normal(loc=mean, scale=std_dev, size=data.shape)
    result = data + noise
    return result.to(data.dtype)


mean1 = []
mean2 = []
mean3 = []
sd1 = []
sd2 = []
sd3 = []

path = "/home/schilak/TrainingData"
for group in os.listdir(path):
    for label in os.listdir(path + "/" + group):
        if label != ".DS_Store":
            for image in os.listdir(path + "/" + group + "/" + label):

                img_path = path + "/" + group + "/" + label + "/" + image
                # img_path = "C:/PhD/ES Competition/sample_code/TrainingSet/G10/NV/8889058.jpg"
                print(img_path)
                
                train_data = MyDataset(group, "train", [(img_path, label, 1)], None)
                train_loader = DataLoader(train_data)
                for i, (images, targets, G, iteration) in enumerate(train_loader):
                    print(images)

                    # convert this image to numpy array
                    img_normalized = np.array(images[0])
                    
                    # transpose from shape of (3,,) to shape of (,,3)
                    img_normalized = img_normalized.transpose(1, 2, 0)

                    print(img_normalized)
                    
                    # display the normalized image
                    plt.imshow(img_normalized)
                    plt.xticks([])
                    plt.yticks([])

                    plt.show()
                    if iteration == 1:
                        #images = add_white_gaussian_noise(images, mean=0.0, std_dev=0.2)
                        # for i1 in range(len(images)):
                        #     images[i1] = add_white_gaussian_noise(images[i1], mean=0.0, std_dev=0.05)
                            # for i2 in range(len(images[i1])):
                                
                                # for i3 in range(len(images[i1][i2])):
                                    
                                #     images[i1][i2][i3] = add_white_gaussian_noise(images[i1][i2][i3], mean=0.0, std_dev=0.05)
                                
                        # convert this image to numpy array
                        img_normalized = np.array(images[0])

                        print(images)
                        
                        # transpose from shape of (3,,) to shape of (,,3)
                        img_normalized = img_normalized.transpose(1, 2, 0)

                        print(img_normalized)
                        
                        # display the normalized image
                        plt.imshow(img_normalized)
                        plt.xticks([])
                        plt.yticks([])

                        plt.show()
                # break


                img = Image.open(img_path)

                # normalize = transforms.Normalize(mean=[0.6424, 0.5305, 0.4971],
                #                          std=[0.0418, 0.0533, 0.0648])
                normalize = transforms.Normalize(mean=[0.6015, 0.4501, 0.4230],
                                         std=[0.3394, 0.2834, 0.2840])
                transform = transforms.Compose([
                    transforms.Resize(224),  # 256
                    transforms.CenterCrop(224),  # 224
                    # transforms.CenterCrop(500),  # min width = 450, min height = 576
                    transforms.ToTensor(),
                    # normalize,
                ])

                img_normalized = transform(img)
                
                # convert this image to numpy array
                img_normalized = np.array(img_normalized)
                
                # transpose from shape of (3,,) to shape of (,,3)
                img_normalized = img_normalized.transpose(1, 2, 0)

                print(img_normalized)
                
                # display the normalized image
                plt.imshow(img_normalized)
                plt.xticks([])
                plt.yticks([])

                plt.show()
                
                
                # transform the pIL image to tensor
                # image
                # img_tr = transform(img)
                # mean, std = img_tr.mean([1,2]), img_tr.std([1,2])

                # mean1.append(img_tr.mean([1,2])[0])
                # mean2.append(img_tr.mean([1,2])[1])
                # mean3.append(img_tr.mean([1,2])[2])

                # sd1.append(img_tr.std([1,2])[0])
                # sd2.append(img_tr.std([1,2])[1])
                # sd3.append(img_tr.std([1,2])[2])

                # print("mean and std before normalize:")
                # print("Mean of the image:", mean)
                # print("Std of the image:", std)

                # break
                
        break
    break

print(np.mean(mean1))
print(np.mean(mean2))
print(np.mean(mean3))
print(np.mean(sd1))
print(np.mean(sd2))
print(np.mean(sd3))
                
                


# # load the image
# img_path = 'TrainingSet/G7/VASC/0061557.jpg'
# img = Image.open(img_path)


# normalize = transforms.Normalize(mean=[0.6551, 0.5194, 0.5850],
#                                          std=[0.1044, 0.1505, 0.1302])

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])


 
# Convert tensor image to numpy array
# img_np = np.array(img_tr)
 
# plot the pixel values
# plt.hist(img_np.ravel(), bins=50, density=True)
# plt.xlabel("pixel values")
# plt.ylabel("relative frequency")
# plt.title("distribution of pixels")

# img_normalized = transform(img)
 
# # convert this image to numpy array
# img_normalized = np.array(img_normalized)
 
# # transpose from shape of (3,,) to shape of (,,3)
# img_normalized = img_normalized.transpose(1, 2, 0)
 
# # display the normalized image
# plt.imshow(img_normalized)
# plt.xticks([])
# plt.yticks([])

# plt.show()