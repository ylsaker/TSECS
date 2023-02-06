#-*-coding:utf-8-*-
import numpy as np
import os
import random
from tqdm import tqdm
import numpy as np
from concurrent import futures
import torch
import torchvision.transforms as transforms
from PIL import Image
from utils import *
import matplotlib.pyplot as plt
np.random.seed(0)
random.seed(0)


def readcsv(filename, image_PATH, data_source, train_or_test):
    """
    mini-Imagenet: filename==.csv
    officehome: filename=='train' or 'test'

    """
    # data_source = FLAGS.data_source
    if data_source == 'mini-Imagenet':
        filename = [f for f in filename if train_or_test in f]
        if len(filename) == 0 or len(filename) >1:
            raise ValueError("Wrong split .csv path.")
        csv = open(filename[0], 'r')
        nameList = []
        for line in csv:
            line = line.strip('\n')
            _, label = line.split(',')
            if label[0] == 'n':
                nameList.append(label)
        csv.close()
        nameList = set([fl for fl in nameList])
        # print(nameList)
        # print(len(nameList))
        return nameList
    elif data_source == 'officehome':
        def get_category(officehome_path):
            class_name = os.listdir(os.path.join(officehome_path, 'Clipart'))
            # print(class_name)
            # print(len(class_name))
            return class_name
        def select_trts(officehome_path):
            class_name = get_category(officehome_path)
            class_idx = list(range(len(class_name)))
            random.shuffle(class_idx)
            train_class = ['Speaker', 'Shelf', 'Backpack', 'Exit_Sign', 'Batteries', 'Bike', 'Chair', 'Notebook', 'Eraser',
             'Push_Pin', 'Pen', 'Fan', 'Sink', 'Mouse', 'Bucket', 'Knives', 'Laptop', 'Curtains', 'Marker',
             'Pan', 'ToothBrush', 'Drill', 'TV', 'Table', 'Couch', 'Radio', 'Folder',
             'Kettle', 'Refrigerator', 'Webcam', 'Candles', 'Telephone', 'Soda', 'Calendar', 'Alarm_Clock', 'Ruler',
             'Toys', 'Mug', 'Glasses', 'Flowers']
            val_class = ['Flipflops', 'Printer',  'Hammer', 'Pencil', 'Lamp_Shade', 'Trash_Can',  'Mop', 'Computer', 'Postit_Notes', 'Spoon']
            test_class = ['Calculator', 'Desk_Lamp', 'Oven', 'Sneakers', 'Paper_Clip', 'Fork', 'Helmet', 'Monitor', 'Keyboard', 'Bed', 'Bottle',
                          'Screwdriver',  'Scissors', 'File_Cabinet',  'Clipboards',]

            assert len(train_class)+len(val_class)+len(test_class) == len(class_name)
            print(len(train_class), len(val_class), len(test_class))
            return {'train': train_class, 'val': val_class, 'test': test_class}

        return select_trts(image_PATH)[train_or_test]
    elif data_source == 'visda':
        train_class = ['stethoscope', 'carrot', 'lobster', 'grass', 'flashlight', 'calendar', 'spreadsheet', 'sun',
                       'nail', 'cello', 'paint_can', 'house', 'sink', 'saw', 'owl', 'keyboard', 'nose', 'suitcase',
                       'pillow', 'ant', 'spoon', 'firetruck', 'scorpion', 'mug', 'rake', 'The_Great_Wall_of_China',
                       'power_outlet', 'bandage', 'trombone', 'pizza', 'backpack', 'goatee', 'traffic_light', 'bowtie',
                       'pond', 'mushroom', 'bee', 'diving_board', 'barn', 'fire_hydrant', 'flamingo', 'butterfly',
                       'teddy-bear', 'police_car', 'snake', 'potato', 'beard', 'snowflake', 'radio', 'scissors',
                       'river', 'alarm_clock', 'marker', 'stop_sign', 'helicopter', 'jacket', 'whale', 'fence', 'frog',
                       'passport', 'knee', 'ice_cream', 'helmet', 'hockey_stick', 'raccoon', 'shovel', 'donut',
                       'parrot', 'flying_saucer', 'peanut', 'angel', 'book', 'The_Eiffel_Tower', 'onion', 'cloud',
                       'mouse', 'ambulance', 'lightning', 'campfire', 'toothpaste', 'flower', 'toaster', 'motorbike',
                       'circle', 'dog', 'clarinet', 'squiggle', 'elephant', 'garden_hose', 'boomerang', 'lion',
                       'watermelon', 'hedgehog', 'popsicle', 'bottlecap', 'coffee_cup', 'dumbbell', 'skateboard',
                       'tooth', 'ocean', 'crab', 'bulldozer', 'eye', 'door', 'tiger', 'frying_pan', 'wine_bottle',
                       'hockey_puck', 'peas', 'lollipop', 'dishwasher', 'broccoli', 'bench', 'chair', 'garden',
                       'speedboat', 'birthday_cake', 'school_bus', 'candle', 'strawberry', 'squirrel', 'snorkel',
                       'duck', 'castle', 'toilet', 'cup', 'mermaid', 'telephone', 'hexagon', 'ear', 'moustache',
                       'hurricane', 'shark', 'piano', 'hamburger', 'pliers', 'harp', 'bucket', 'shoe', 'car', 'teapot',
                       'screwdriver', 'purse', 'fork', 'lighter', 'horse', 'tornado', 'blueberry', 'laptop', 'banana',
                       'house_plant', 'bathtub', 'necklace', 'rollerskates', 'giraffe', 'bread', 'vase', 'bird',
                       'hammer', 'canoe', 'dolphin', 'cannon', 'leg', 'zebra', 'drums', 'bat', 'sheep', 'beach', 'arm',
                       'paintbrush', 'cactus', 'church', 'steak', 'stairs', 'trumpet', 'truck', 'airplane', 'anvil',
                       'map', 'brain', 'baseball', 'lantern', 'umbrella', 'camel', 'rainbow', 'knife', 'lighthouse',
                       'yoga', 'rhinoceros', 'megaphone', 'submarine', 'cell_phone', 'bracelet', 'elbow', 'wine_glass',
                       'axe', 'sword', 'streetlight', 'soccer_ball', 'hot_dog', 'shorts', 'hot_air_balloon',
                       'pickup_truck', 'crown', 'matches', 'hat', 'triangle', 'pear', 'chandelier', 'sandwich',
                       'skyscraper', 'camouflage', 'mouth', 'bus', 'cake', 'hot_tub', 'bicycle']
        test_class = ['skull', 'key', 'cat', 'moon', 'tree', 'hourglass', 'basket', 'rain', 'mountain', 'snowman',
                      'sweater', 'crocodile', 'ladder', 'sailboat', 'envelope', 'grapes', 'television', 'rabbit',
                      'dragon', 'animal_migration', 'clock', 'ceiling_fan', 'picture_frame', 'snail', 'jail',
                      'eyeglasses', 'golf_club', 'star', 'baseball_bat', 'spider', 'cruise_ship', 'postcard', 'bed',
                      'bridge', 'tent', 'guitar', 'octopus', 'face', 'parachute', 'hand', 'apple', 'pineapple',
                      'sea_turtle', 'train', 'finger', 'violin', 'pig', 'see_saw']
        val_class = ['microphone', 'binoculars', 'basketball', 'couch', 'square', 'remote_control', 'headphones',
                     'camera', 'broom', 'The_Mona_Lisa', 'roller_coaster', 'bush', 'sock', 'drill', 'flip_flops',
                     'aircraft_carrier', 'swan', 'smiley_face', 'pool', 'palm_tree', 'swing_set', 'lipstick', 'table',
                     'rifle', 'panda', 'penguin', 'windmill', 'bear', 'feather', 'paper_clip', 'pencil', 'kangaroo',
                     'mosquito', 'zigzag', 'saxophone', 'fish', 'cookie', 'tractor', 'asparagus', 'string_bean', 'foot',
                     'hospital', 'monkey']

        sets = {'train': train_class, 'val': val_class, 'test': test_class}
        return sets[train_or_test]
    elif data_source == 'tiered-Imagenet':
        root_path = os.path.join(image_PATH, 'photo')
        all_classes = os.listdir(root_path)
        train_class = [p for p in all_classes if 'train' in p]
        val_class = [p for p in all_classes if 'val' in p]
        test_class = [p for p in all_classes if 'test' in p]
        sets = {'train': train_class, 'val': val_class, 'test': test_class}
        return sets[train_or_test]




def shuffle_datas(x, y, m):
    idxs = list(range(len(x)))
    random.shuffle(idxs)
    tx = [x[i] for i in idxs]
    ty = [y[i] for i in idxs]
    tm = [m[i] for i in idxs]
    return tx, ty, tm

def uda_random_box(data_model, raw_class_num, m):
    c = random.randint(0, raw_class_num-1)
    idx = random.randint(0, len(data_model[m][c]) - 1)
    return data_model[m][c][idx]


def sample_task(data_model, raw_class_num, model, query_num_per_class_per_model=1, class_num=5,
                    support_num_per_class_per_model=1, train_or_test='train', random_target=False):
    """
    :param data_model: list of module name e.g. ['art_painting', 'cartoon', 'photo', 'sketch']
    :param class_num: n-ways.
    :return: list of dict of path and label e.g [{'support_xy': [['/data2/hsq/Project/metric_PACS/pacs_filename/art_painting/horse/pic_072.jpg'], [1]],
                                                'q_xy': [['/data2/hsq/Project/metric_PACS/pacs_filename/art_painting/horse/pic_091.jpg'], [1]]]

    """

    
    classes = random.sample(range(raw_class_num), class_num)
    task = {'x': {'support': [], 'query': []},
            'label': {'support': [], 'query': []},
            'modal': {'support': [], 'query': []},
            'global_label':{'support':[], 'query':[]}}
    for m_label, m in enumerate(model):
        modal_task = {'support': {'data': [], 'label': [], 'modal': []}, 'query': {'data': [], 'label': [], 'modal': []}, }
        for i, c in enumerate(classes):
            idxs = random.sample(range(len(data_model[m][c])), support_num_per_class_per_model)
            support_x = [data_model[m][c][id] for id in idxs]
            support_y = [i for _ in range(support_num_per_class_per_model)]
            support_modal = [m_label for _ in range(support_num_per_class_per_model)]

            if train_or_test != 'train' or m_label == 0:
                query_x = [dm for dm in data_model[m][c] if dm not in support_x]
                # print(query_x, query_num_per_class_per_model)
                query_x = random.sample(query_x, query_num_per_class_per_model)
                query_y = [i for _ in range(query_num_per_class_per_model)]
            else:
                if random_target:
                    query_x = [uda_random_box(data_model, raw_class_num, m) for _ in range(query_num_per_class_per_model)]
                else:
                    query_x = [dm for dm in data_model[m][c] if dm not in support_x]
                    query_x = random.sample(query_x, query_num_per_class_per_model)
                query_y = [-1 for _ in range(query_num_per_class_per_model)]
            query_modal = [m_label for _ in range(query_num_per_class_per_model)]
            modal_task['support']['data'].extend(support_x)
            modal_task['support']['label'].extend(support_y)
            modal_task['support']['modal'].extend(support_modal)
            modal_task['query']['data'].extend(query_x)
            modal_task['query']['label'].extend(query_y)
            modal_task['query']['modal'].extend(query_modal)

            task['global_label']['support'].extend([c for _ in range(support_x.__len__())])
            task['global_label']['query'].extend([c for _ in range(query_x.__len__())])


        task['x']['support'].append(modal_task['support']['data'])
        task['x']['query'].append(modal_task['query']['data'])
        task['label']['support'].extend(modal_task['support']['label'])
        task['label']['query'].extend(modal_task['query']['label'])
        task['modal']['support'].extend(modal_task['support']['modal'])
        task['modal']['query'].extend(modal_task['query']['modal'])
    return task


def split_dataset(model, split_txt, mode, data_source='mini-Imagenet', data_PATH=None):
    """
    split the dataset by modal and class.
    total_group__
                 |_modal1__
                 |         |__c1__all_filenames
                 |         .
                 |         .
                 |         |__c64__......
                 |
                 |_modal2 __
                           |__c1__all_filenames
                           |         .
                           .
                           .
                           |__c64__......

    officehome dir:
                modal:  |__Art
                        |__Clipart
                        |__Product
                        |__RealWorld
                split_dict:
                    train:
                        |__C1
                        |__C2
                        |__C3
                        |__...
                        ...
                        |__C45
                    test:
                        |__C46
                        |__C47
                        |__C48
                        |__...
                        ...
                        |__C60
    mini-Imagenet dir:
                modal:  |__photo___
                        |          |_all files
                        |
                        |__sketch__
                                   |_all files
    """
    train_or_test = mode
    # make sure read the 'train' or 'test' .csv.
    # get the {"path": ../, 'label': 1..} dict for both 2 modals.
    fliename_label = None
    if data_source == 'mini-Imagenet':
        # filename, image_PATH, data_source, train_or_test
        fliename_label = readcsv(split_txt, None, 'mini-Imagenet', train_or_test)
    elif data_source == 'officehome' or 'visda' or 'tiered-Imagenet':
        fliename_label = readcsv(None, data_PATH, data_source, train_or_test)
        # print(fliename_label)
    labels0 = fliename_label
    class_num = len(labels0)
    total_group = {m: [] for m in model}
    # get the modal-label list for unique global label.
    for k, m in enumerate(model):
        model_pic_list = os.listdir(os.path.join(data_PATH, m))
        group = [[] for _ in range(class_num)]
        for i, c in enumerate(labels0):
            if data_source == 'mini-Imagenet':
                group[i] = [os.path.join(m, f) for f in model_pic_list if c in f]
            elif data_source == 'officehome' or 'visda' or 'tiered-Imagenet':
                group[i] = [os.path.join(m, c, f) for f in os.listdir(os.path.join(data_PATH, m, c))]
        total_group[m] = group
    return total_group, class_num



class DataGenerator(object):
    def __init__(self, mode, datasource, imagesize, episode, way_num, support_num, query_num, data_dir, split_PATH, source_domain=None, target_domain=None, random_target=False, opt=None):
        super(DataGenerator, self).__init__()
        self.opt = opt

        self.mode = mode
        self.episode = episode
        self.raw_path = data_dir
        self.img_size = imagesize
        self.datasource = datasource
        self.source_domain = source_domain
        self.target_domain = target_domain
        self.way_num = way_num
        self.support_num, self.query_num = support_num, query_num
        self.input_dim = np.prod(self.img_size)*3
        self.split_txt, self.model, self.train_test = self.construct(datasource, split_PATH)
        self.data_model, self.raw_class_num = split_dataset(self.model, self.split_txt, self.mode, self.datasource,
                                                            self.raw_path)
        self.random_target = random_target
        self.all_tasks = self.generator_tasks()
        self.now_epi = None

    def construct(self, datasource, split_PATH):
        if datasource == 'mini-Imagenet':
            split_path = [os.path.join(split_PATH, t) for t in os.listdir(split_PATH)]
            modal = [self.source_domain, self.target_domain]
            split = ['train', 'test', 'val']
        elif datasource == 'tiered-Imagenet':
            split_path = [os.path.join(split_PATH, t) for t in os.listdir(split_PATH)]
            modal = [self.source_domain, self.target_domain]
            split = ['train', 'test', 'val']
        elif datasource == 'officehome':
            split_path = [os.path.join(split_PATH, t) for t in os.listdir(split_PATH)]
            modal = [self.source_domain, self.target_domain]
            split = ['train', 'test']
        elif datasource == 'visda':
            # split_path = [os.path.join(split_PATH, t) for t in os.listdir(split_PATH)]
            split_path = None
            if self.source_domain != None:
                modal = [self.source_domain, self.target_domain]
            else:
                modal = ['real', 'sketch']
            split = ['train', 'test', 'val']
        else:
            raise ValueError("no dataset named:%s" %datasource)
        self.data_parser = Parsers(datasource, self.raw_path, self.img_size, self.mode)
        print("modal %s -> %s" % (modal[0], modal[1]))
        return split_path, modal, split

    def generator_tasks(self,):
        """
        :param train: train or not.
        :return: all tasks, composed by dicts e.g.{'support_set': support_set, 'query_set':query_set}
                            which value in dicts is list of data tensors, first element such as support_set[0] is
                            image tensor, the next is label one-hot tensors.
        """
        # print('===========================Generating task filedirs===========================')
        raw_path, split_txt, model, train_test = self.raw_path, self.split_txt, self.model, self.train_test
        if raw_path == None or not os.path.exists(raw_path):
            raise ValueError("No such kind of data source: %s"%self.datasource)
        tasknum = self.episode
        all_tasks = []
        for _ in (range(tasknum)):
            task = sample_task(self.data_model, self.raw_class_num, model, query_num_per_class_per_model=self.query_num,
                                     class_num=self.way_num,
                                     support_num_per_class_per_model=self.support_num, train_or_test=self.mode, random_target=self.random_target)
            assert ('x' in task and 'label' in task and 'modal' in task)
            all_tasks.append(task)

            # ==========保存图片文件名========== #
            # if (self.opt is not None) and (
            #         self.opt.save_epoch_item == 0 or self.opt.save_epoch_item == self.opt.epochs - 1):
            if self.opt is not None and self.opt.save_epoch_item >= 0 and self.opt.save_img_label_flag:  # 这里判断不了余数, 没有asp_count加一操作
                # root = '/home/lyz/NAS/lyz/pyProject/experiment_results/TSFL_Broadcast/save_para/'
                root = self.opt.save_para_img_path
                if self.opt.only_test_flag:  # 只测试时保存地址
                    root = self.opt.save_test_para_img_path
                # 新建三个文件夹IMSE(metric)、mode_type、para_name
                metric_path = os.path.join(root, f'{self.opt.metric}')
                if not os.path.exists(metric_path):
                    os.makedirs(metric_path)
                epoch_path = os.path.join(metric_path, f'{self.opt.save_epoch_item}')
                if not os.path.exists(epoch_path):
                    os.makedirs(epoch_path)
                mode_path = os.path.join(epoch_path, f'{self.opt.mode_type}')
                if not os.path.exists(mode_path):
                    os.makedirs(mode_path)
                para_path = os.path.join(mode_path, 'image_lable')
                if not os.path.exists(para_path):
                    os.makedirs(para_path)
                support_path = os.path.join(para_path, 'support_task')
                if not os.path.exists(support_path):
                    os.makedirs(support_path)
                query_path = os.path.join(para_path, 'query_task')
                if not os.path.exists(query_path):
                    os.makedirs(query_path)
                # print(type(imgs))
                saved_support_imgs_name = np.array(task['x']['support'])
                saved_query_imgs_name = np.array(task['x']['query'])
                np.save(os.path.join(support_path, f'{_}.npy'), saved_support_imgs_name)  # key=support/query, 保存为.npy格式
                np.save(os.path.join(query_path, f'{_}.npy'), saved_query_imgs_name)  # key=support/query, 保存为.npy格式
            # ==========保存图片文件名========== #

        return all_tasks

    def __len__(self):
        return len(self.all_tasks)

    def __getitem__(self, index):
        '''
            Load an episode each time, including C-way K-shot and Q-query
        '''

        episode_files = self.all_tasks[index]
        # self.now_epi = episode_files


        task = self.data_parser.make_set_tensor_multi_thread2(episode_files)
        # query_images = np.reshape(np.transpose(task[0]['query'], [0, 1, 4, 2, 3]), [len(self.model), -1, 3, self.img_size, self.img_size])
        # print(task[0]['query'].permute(0, 1, 4, 2, 3).size())
        # print(task[0]['query'].size())
        query_images = torch.reshape(task[0]['query'], [-1, 3, self.img_size, self.img_size])

        query_targets = task[1]['query']
        query_modal = task[2]['query']

        support_images = torch.stack([torch.reshape(task[0]['support'][0], [self.way_num, -1, 3, self.img_size, self.img_size]),
                          torch.reshape(task[0]['support'][1], [self.way_num, -1, 3, self.img_size, self.img_size])], 0)

        support_targets = np.reshape(task[1]['support'], [2, -1])[0]
        query_global_targets = np.reshape(task[3]['query'], [2, -1])[0]
        return query_images, query_targets, query_modal, support_images, support_targets, query_global_targets


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class Parsers:
    def __init__(self, datasource, data_PATH, img_size, mode):
        if mode == 'val' or mode == 'test':

            image_size = img_size
            if datasource == 'officehome':
                self.transform = transforms.Compose([
                    transforms.Resize([92, 92]),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor()])
            else:
                self.transform = transforms.Compose([

                    transforms.Resize([image_size, image_size]),

                    # transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                         np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
                ]
                )



        elif mode == 'train':
            image_size = img_size
            tran_list = []
            if datasource == 'officehome':
                tran_list.append(transforms.RandomRotation(degrees=(-10, 10)))
                tran_list.append(transforms.RandomResizedCrop(image_size, scale=(0.7, 1)))
            else:
                tran_list.append(transforms.RandomResizedCrop(image_size, scale=(0.9, 1)))
            if datasource == 'officehome':
                tran_list.extend([transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])
            else:
                tran_list.extend([transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                                       np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
            self.transform = transforms.Compose(tran_list)



        if datasource == 'tiered-Imagenet':
            # print('tieret-Imagenet transform.')
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])





        if datasource == 'mini-Imagenet' or 'officehome':
            self.modal = 2
            self.data_PATH = data_PATH
            self.img_size = img_size
            self.way_num = 5

        else:
            raise ValueError("no such datasource : %s" % datasource)


    def compose_parser(self, data_list, paser_list, index, key):
        img_idx = (int((index) / len(data_list[0][key][0])), index % len(data_list[0][key][0]))
        d = [data_list[0][key][img_idx[0]][img_idx[1]], data_list[1][key][index], data_list[2][key][index]]
        args = list(zip(d, paser_list))
        ret_data = [self.parse_task_meta_data(a) for a in args]
        return ret_data

    def multi_processing_to_sq(self, inp):
        data_list, paser_list, key = inp
        n = len(data_list[1][key])
        with futures.ThreadPoolExecutor(max_workers=5) as executor:
            future = [executor.submit(self.compose_parser, data_list, paser_list, i, key) for i in range(n)]
            set = [f.result() for f in future]
        return set

    def handle(self, set):
        modals = self.modal
        x = [[] for _ in range(modals)]
        label = []
        modal = []
        for idx, modal_set in enumerate(set):
            for item in modal_set:
                x[idx].append(item[0])
                label.append(item[1])
                modal.append(item[2])
            x[idx] = torch.stack(x[idx], dim=0)
        return torch.stack(x, dim=0), np.array(label), np.array(modal)

    def parser(self, ret):
        modals = self.modal
        # modal_keys = [np.eye(modals)[i] for i in range(modals)]
        modal_keys = [i for i in range(modals)]
        data = [[] for _ in range(modals)]
        index_c = lambda x: [x == mk for mk in (modal_keys)]
        for r in ret:
            i = index_c(r[2]).index(True)
            data[i].append(r)
        return data

    def make_set_tensor_multi_thread(self, dict_set):
        data_list = [dict_set['x'], dict_set['label'], dict_set['modal']]
        paser_list = [self.image_set_parser, self.label_parser, self.modal_parser]
        inp = [(data_list, paser_list, 'support'), (data_list, paser_list, 'query')]
        with futures.ThreadPoolExecutor(max_workers=4) as executor:
            ite = executor.map(self.multi_processing_to_sq, inp)
        ret = [i for i in ite]
        support = self.handle(self.parser(ret[0]))
        query = self.handle(self.parser(ret[1]))
        x = {'support': support[0], 'query': query[0]}
        label = {'support': support[1], 'query': query[1]}
        modal = {'support': support[2], 'query': query[2]}
        return x, label, modal

    def make_set_tensor_multi_thread2(self, dict_set):
        support = self.get_data('support', dict_set)
        query = self.get_data('query', dict_set)
        x = {'support': support[0], 'query': query[0]}
        label = {'support': support[1], 'query': query[1]}
        modal = {'support': support[2], 'query': query[2]}
        global_label = {'support': support[3], 'query': query[3]}
        return x, label, modal, global_label

    def get_data(self, key, dict_set):
        imgs = dict_set['x'][key]
        labels = dict_set['label'][key]
        modals = dict_set['modal'][key]
        global_label = dict_set['global_label'][key]

        img_tensorset = []
        for m in range(len(imgs)):
            imgtensor = []
            for p in imgs[m]:
                imgtensor.append(self.image_set_parser(p))
            img_tensorset.append(torch.stack(imgtensor, 0))

        img_tensorset = torch.stack(img_tensorset, 0)
        # B, C, h, w = img_tensorset.size()
        # print(img_tensorset.size())
        return img_tensorset, np.array(labels), np.array(modals), np.array(global_label)





    def image_set_parser(self, image_filename):
        image_filename = os.path.join(self.data_PATH, image_filename)
        if not os.path.exists(image_filename):
            raise ValueError("no such file: %s." % image_filename)
        # img = cv2.resize(cv2.imread(image_filename), (self.img_size, self.img_size))
        img = pil_loader(image_filename)
        # pil_im = Image.open('1.jpg').convert('L') #灰度操作
        img = self.transform(img)
        return img

    def parse_task_meta_data(self, d_parser):
        d, parser = d_parser
        ret = parser(d)
        return ret

    def label_parser(self, label_int):
        n_ways = self.way_num
        # onehot_label = np.eye(n_ways)[label_int]
        onehot_label = label_int
        return onehot_label

    def modal_parser(self, modal):
        m_modal = self.modal
        # onehot_modal = np.eye(m_modal)[modal]
        onehot_modal = modal
        return onehot_modal








