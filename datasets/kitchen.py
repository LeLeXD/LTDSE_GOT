from __future__ import absolute_import, print_function, unicode_literals

import os
import glob
import numpy as np
import io
import six
from itertools import chain

class kitchen(object): 
    __data20 = [
    'P03_P03_20_verb-4_noun-9_action-1_id-440', 
    'P05_P05_03_verb-2_noun-35_action-1262_id-16', 
    'P05_P05_08_verb-7_noun-19_action-1_id-8', 
    'P06_P06_03_verb-7_noun-19_action-1_id-39', 
    'P07_P07_04_verb-4_noun-3_action-1750_id-41',                  
    'P07_P07_09_verb-20_noun-29_action-1_id-3', 
    'P12_P12_01_verb-1_noun-12_action-1_id-697', 
    'P12_P12_07_verb-4_noun-20_action-1729_id-475', 
    'P13_P13_08_verb-2_noun-13_action-1_id-116', 
    'P13_P13_09_verb-2_noun-35_action-1262_id-133'
                ]

    __version_dict = {
        2020: __data20
                }

    def __init__(self, root_dir, version=2020):
        super(kitchen, self).__init__()
        assert version in self.__version_dict

        self.root_dir = root_dir
        self.version = version        
        self._check_integrity(root_dir, version)

        valid_seqs = self.__version_dict[version]
        self.anno_files = sorted(list(chain.from_iterable(glob.glob(
            os.path.join(root_dir, s, 'groundtruth*.txt')) for s in valid_seqs)))
       
        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
        self.seq_names = [os.path.basename(d) for d in self.seq_dirs]


    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(
            os.path.join(self.seq_dirs[index], 'img/*.jpg')))
        img_files = np.array(img_files)
        # special sequences
        # (visit http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html for detail)
        seq_name = self.seq_names[index]
        if seq_name == "P05_P05_03_verb-2_noun-35_action-1262_id-16":            
            img_files = img_files[np.r_[0:468-1, 477-1:631-1, 900-1:1610-1]]
        elif seq_name == 'P05_P05_08_verb-7_noun-19_action-1_id-8':
            img_files = img_files[9-1:380-1]
        elif seq_name == 'P06_P06_03_verb-7_noun-19_action-1_id-39':
            img_files = img_files[40-1:590-1]
        elif seq_name == 'P07_P07_04_verb-4_noun-3_action-1750_id-41':
            img_files = img_files[0:416-1]
        elif seq_name == 'P07_P07_09_verb-20_noun-29_action-1_id-3':            
            img_files = img_files[np.r_[0:640-1, 661-1:740-1]]
        elif seq_name == 'P12_P12_01_verb-1_noun-12_action-1_id-697':
            img_files = img_files[0:504-1]
        elif seq_name == 'P12_P12_07_verb-4_noun-20_action-1729_id-475':            
            img_files = img_files[np.r_[0:383-1, 491-1:680-1]]
        elif seq_name == 'P13_P13_09_verb-2_noun-35_action-1262_id-133':            
            img_files = img_files[np.r_[0:25-1, 66-1:1460-1]]

        # to deal with different delimeters
        with open(self.anno_files[index], 'r') as f:
            anno = np.loadtxt(io.StringIO(f.read().replace(',', ' ')))           
        assert len(img_files) == len(anno)
        assert anno.shape[1] == 4

        return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _check_integrity(self, root_dir, version):
        assert version in self.__version_dict
        seq_names = self.__version_dict[version]

        if os.path.isdir(root_dir) and len(os.listdir(root_dir)) > 0:
            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset not found or corrupted. ' +
                            'You can use download=True to download it.')