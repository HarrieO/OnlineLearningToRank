# -*- coding: utf-8 -*-

import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.dataset import DataSet


def get_datasets(sim_args):
    """
    Function for retrieving datasets from simulation arguments.
    """
    if len(sim_args.data_sets) == 1 and sim_args.data_sets[0] == 'all':
        data_tags = [
            'Webscope_C14_Set1',
            #'Webscope_C14_Set2',
            'MSLR-WEB10k',
            'MQ2007',
            'MQ2008',
            'OHSUMED',
            'NP2003',
            'NP2004',
            'HP2003',
            'HP2004',
            'TD2003',
            'TD2004',
            ]
    elif len(sim_args.data_sets) == 1 and sim_args.data_sets[0] == 'cikm2018':
        data_tags = [
            'Webscope_C14_Set1',
            #'Webscope_C14_Set2',
            'MSLR-WEB10k',
            'MQ2007',
            'MQ2008',
            # 'mnist',
            ]
        # random.shuffle(data_tags)
    elif len(sim_args.data_sets) == 1 and sim_args.data_sets[0] == 'thesis':
        data_tags = [
            'Webscope_C14_Set1',
            #'Webscope_C14_Set2',
            'MSLR-WEB10k',
            ]
        # random.shuffle(data_tags)
    elif len(sim_args.data_sets) == 1 and sim_args.data_sets[0] == 'small':
        data_tags = [
            'NP2003',
            'NP2004',
            'HP2003',
            'HP2004',
            'TD2003',
            'TD2004',
            'MQ2007',
            'MQ2008',
            'OHSUMED',
            ]
    elif len(sim_args.data_sets) == 1 and sim_args.data_sets[0] == 'small1':
        data_tags = [
            'NP2003',
            'NP2004',
            'HP2003',
            'HP2004',
            'TD2003',
            ]
    elif len(sim_args.data_sets) == 1 and sim_args.data_sets[0] == 'small2':
        data_tags = [
            'TD2004',
            'MQ2007',
            'MQ2008',
            'OHSUMED',
            ]
    elif len(sim_args.data_sets) == 1 and sim_args.data_sets[0] == 'letor64':
        data_tags = [
            'NP2003',
            'NP2004',
            'HP2003',
            'HP2004',
            'TD2003',
            'TD2004',
            ]
        # random.shuffle(data_tags)
    else:
        data_tags = sim_args.data_sets
    for data_tag in data_tags:
        assert data_tag in DATASET_COLLECTION, 'Command line input is currently not supported.'
        yield DATASET_COLLECTION[data_tag]


DATASET_COLLECTION = {}
DATASET_COLLECTION['NP2003'] = DataSet('2003_np', '/zfs/ilps-plex1/slurm/datastore/hooster2/datasets/2003_np_dataset/Fold*/',
                                       'bin', True, 59,
                                       multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(26,41), #LMIR
                                               [41,42],      #SiteMap
                                               [49,50]       #HITS
                                             ])
DATASET_COLLECTION['NP2004'] = DataSet('2004_np', '/zfs/ilps-plex1/slurm/datastore/hooster2/datasets/2004_np_dataset/Fold*/',
                                       'bin', True, 64,
                                       multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(26,41), #LMIR
                                               [41,42],      #SiteMap
                                               [49,50]       #HITS #19 total
                                             ])
DATASET_COLLECTION['HP2003'] = DataSet('2003_hp', '/zfs/ilps-plex1/slurm/datastore/hooster2/datasets/2003_hp_dataset/Fold*/',
                                       'bin', True, 59,
                                       multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(26,41), #LMIR
                                               [41,42],      #SiteMap
                                               [49,50]       #HITS
                                             ])
DATASET_COLLECTION['HP2004'] = DataSet('2004_hp', '/zfs/ilps-plex1/slurm/datastore/hooster2/datasets/2004_hp_dataset/Fold*/',
                                       'bin', True, 64,
                                       multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(26,41), #LMIR
                                               [41,42],      #SiteMap
                                               [49,50]       #HITS
                                             ])
DATASET_COLLECTION['TD2003'] = DataSet('2003_td', '/zfs/ilps-plex1/slurm/datastore/hooster2/datasets/2003_td_dataset/Fold*/',
                                       'bin', True, 59,
                                       multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(26,41), #LMIR
                                               [41,42],      #SiteMap
                                               [49,50]       #HITS
                                             ])
DATASET_COLLECTION['TD2004'] = DataSet('2004_td', '/zfs/ilps-plex1/slurm/datastore/hooster2/datasets/2004_td_dataset/Fold*/',
                                       'bin', True, 64,
                                       multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(26,41), #LMIR
                                               [41,42],      #SiteMap
                                               [49,50]       #HITS
                                             ])

DATASET_COLLECTION['MQ2008'] = DataSet('MQ2008', '/zfs/ilps-plex1/slurm/datastore/hooster2/datasets/MQ2008/Fold*/', 'short',
                                       True, 40,
                                       multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(26,41)  #LMIR #25 total
                                             ])
DATASET_COLLECTION['MQ2007'] = DataSet('MQ2007', [
                                '/zfs/ilps-plex1/slurm/datastore/hooster2/datasets/MQ2007/Fold*/',
                                '/Users/hroosterhuis/ILPS/datasets/MQ2007/Fold*/'
                                ], 'short',
                                       True, 41,
                                       multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(26,41)  #LMIR
                                             ])
DATASET_COLLECTION['OHSUMED'] = DataSet('OHSUMED', '/zfs/ilps-plex1/slurm/datastore/hooster2/datasets/OHSUMED/Fold*/', 'short'
                                        , True, 36,
                                       multileave_feat=[
                                               #[6,7],       #HITS
                                               range(9,13), #TF-IDF
                                               [28],        #sitemap
                                               range(15,28), #BM25 and LMIR
                                             ])

DATASET_COLLECTION['MSLR-WEB10k'] = DataSet('MSLR-WEB10k',
                                            '/zfs/ilps-plex1/slurm/datastore/hooster2/datasets/MSLR-WEB10k/Fold*/', 'long',
                                            False, 136,
                                            multileave_feat=[
                                               range(71,91), #TF-IDF
                                               range(106,111), #BM25
                                               range(111,126), #LMIR # 40 total
                                               # range(96,106), #Boolean Model, Vector Space Model
                                             ])
DATASET_COLLECTION['MSLR-WEB30k'] = DataSet('MSLR-WEB30k',
                                            '/zfs/ilps-plex1/slurm/datastore/hooster2/datasets/MSLR-WEB30k/Fold*/', 'long',
                                            False, 136, max_folds=2,
                                            multileave_feat=[
                                               range(71,91), #TF-IDF
                                               range(106,111), #BM25
                                               range(111,126), #LMIR
                                               # range(96,106), #Boolean Model, Vector Space Model
                                             ])

DATASET_COLLECTION['Webscope_C14_Set1'] = DataSet('Webscope_C14_Set1',
                                                  '/zfs/ilps-plex1/slurm/datastore/hooster2/datasets/Webscope_C14_Set1/',
                                                  'long', False, 471,
                                                  multileave_feat=[
                                                     [1, 2, 6, 7, 8, 9, 10, 11, 12, 17, 18, 20, 21, 23, 26, 27, 28, 29, 30, 31, 32, 34, 36, 37, 39, 41, 43, 44, 45, 46, 48, 53, 55, 56, 58, 60, 62, 64, 66, 69, 70, 71, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 85, 86, 88, 89, 91, 96, 97, 98, 99, 100, 101, 102, 104, 107, 108, 110, 111, 114, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 131, 133, 135, 137, 138, 139, 140, 141, 143, 144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 155, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 172, 173, 174, 175, 176, 177, 178, 179, 181, 182, 186, 187, 189, 190, 191, 192, 193, 195, 196, 197, 201, 202, 204, 205, 208, 212, 215, 216, 219, 220, 222, 223, 224, 225, 226, 227, 229, 230, 231, 232, 233, 234, 235, 236, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 253, 254, 255, 256, 257, 260, 261, 264, 265, 266, 267, 268, 271, 274, 275, 276, 277, 279, 281, 282, 283, 284, 285, 286, 287, 289, 290, 292, 294, 295, 297, 298, 299, 300, 301, 302, 304, 305, 309, 311, 312, 313, 317, 320, 321, 322, 323, 324, 325, 326, 329, 330, 331, 332, 335, 337, 338, 339, 341, 342, 344, 345, 347, 348, 349, 350, 352, 353, 355, 356, 358, 359, 361, 362, 363, 364, 366, 369, 372, 374, 376, 377, 378, 379, 381, 382, 383, 384, 385, 387, 389, 390, 393, 394, 395, 397, 398, 399, 400, 404, 405, 408, 410, 412, 414, 416, 417, 418, 421, 426, 427, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 450, 451, 452, 453, 454, 456, 457, 458, 459, 461, 463, 465, 468, 470, 472, 473, 474, 475, 476, 477, 480, 481, 483, 485, 486, 487, 488, 489, 492, 494, 495, 497, 498, 499, 500, 502, 504, 505, 507, 508, 511, 513, 514, 515, 518, 519, 521, 525, 527, 528, 529, 531, 532, 533, 534, 535, 537, 538, 539, 540, 541, 542, 544, 545, 546, 548, 550, 554, 556, 557, 558, 561, 562, 563, 564, 565, 566, 568, 569, 570, 571, 572, 574, 575, 578, 579, 580, 581, 583, 585, 586, 587, 589, 592, 594, 595, 596, 598, 600, 602, 603, 604, 605, 606, 607, 608, 610, 611, 612, 613, 614, 615, 618, 620, 621, 622, 623, 624, 625, 627, 628, 631, 633, 634, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 647, 648, 650, 654, 656, 657, 658, 659, 660, 661, 663, 664, 665, 666, 669, 671, 674, 676, 677, 678, 680, 682, 683, 685, 686, 687, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699]
                                                  ])
DATASET_COLLECTION['Webscope_C14_Set2'] = DataSet('Webscope_C14_Set2',
                                                  '/zfs/ilps-plex1/slurm/datastore/hooster2/datasets/Webscope_C14_Set2/',
                                                  'long', False, 592)
DATASET_COLLECTION['istella'] = DataSet('istella',
                                                  '/zfs/ilps-plex1/slurm/datastore/hooster2/datasets/istella/',
                                                  'long', False, 220)
DATASET_COLLECTION['mnist'] = DataSet('mnist',
                                             '/zfs/ilps-plex1/slurm/datastore/hooster2/datasets/MNIST/Fold*/',
                                             'long', False, 784)


DATASET_COLLECTION['local_single_NP2003'] = DataSet('2003_np',
                                             [
                                                '/Users/hroosterhuis/Documents/ILPS/datasets/NP2003/Fold1/',
                                                '/Users/hroosterhuis/ILPS/datasets/NP2003/Fold1/'
                                             ],
                                             'bin', True, 59,
                                             multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(36,41), #LMIR
                                               [41,42],      #SiteMap
                                               [49,50]       #HITS
                                             ])
DATASET_COLLECTION['local_NP2003'] = DataSet('2003_np',
                                             '/Users/hroosterhuis/Documents/ILPS/datasets/NP2003/Fold*/'
                                             , 'bin', True, 59,
                                             multileave_feat=[
                                               range(11,16), #TF-IDF
                                               range(21,26), #BM25
                                               range(36,41), #LMIR
                                               [41,42],      #SiteMap
                                               [49,50]       #HITS
                                             ])
DATASET_COLLECTION['local_MQ2008'] = DataSet('MQ2008',
                                             '/Users/hroosterhuis/Documents/ILPS/datasets/MQ2008/Fold*/'
                                             , 'short', True, 40)


DATASET_COLLECTION['local_single_MNIST'] = DataSet('mnist',
                                             [
                                                '/Users/hroosterhuis/ILPS/datasets/LTRMNIST/Fold1/'
                                             ],
                                             'long', False, 784,)
