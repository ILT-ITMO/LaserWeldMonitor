from typing import Tuple, Literal, Union
import json
import pandas as pd
import numpy as np


def prepare_dataset(path: str, class_id: str = 'hi', 
                    type: Literal['reduced', 'lstm'] = 'reduced') -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]:
    with open(path, 'r') as f:
        data = json.load(f)

    quality_thresholds = [0.15, 0.3, 0.65, 0.65, 0.3, 0.3, 0]

    quality = {"thermogram_7.npy": 
    [[0.091, 0.078, 0.000, 0.132, 0.091, 0.080, 0.000],
    [0.000, 0.132, 0.000, 0.080, 0.134, 0.019, 0.000],
    [0.000, 0.106, 0.000, 0.035, 0.329, 0.037, 0.000],
    [0.000, 0.107, 0.000, 0.032, 0.143, 0.088, 0.000]],
    "thermogram_8.npy":
    [[0.000, 0.126, 0.000, 0.000, 0.147, 0.104, 0.000],
    [0.000, 0.158, 0.000, 0.000, 0.242, 0.027, 0.000],
    [0.000, 0.104, 0.000, 0.000, 0.245, 0.073, 0.000],
    [0.000, 0.104, 0.000, 0.094, 0.239, 0.100, 0.000]],
    "thermogram_9.npy":
    [[0.000, 0.000, 0.000, 0.244, 0.121, 0.046, 0.000],
    [0.000, 0.089, 0.000, 0.000, 0.161, 0.046, 0.000],
    [0.000, 0.049, 0.000, 0.000, 0.129, 0.159, 0.000],
    [0.000, 0.082, 0.000, 0.000, 0.114, 0.221, 0.000]],
    "thermogram_10.npy":
    [[0.000, 0.100, 0.000, 0.000, 0.283, 0.055, 0.000],
    [0.000, 0.189, 0.000, 0.000, 0.264, 0.049, 0.000],
    [0.000, 0.109, 0.000, 0.000, 0.102, 0.174, 0.000],
    [0.000, 0.082, 0.078, 0.078, 0.053, 0.217, 0.000],
    ],
    "thermogram_11.npy":
    [[0.066, 0.067, 0.029, 0.109, 0.000, 0.066, 0.000],
    [0.194, 0.196, 0.000, 0.000, 0.000, 0.040, 0.000],
    [0.000, 0.138, 0.000, 0.136, 0.151, 0.032, 0.000],
    [0.090, 0.075, 0.000, 0.046, 0.000, 0.149, 0.000],
    ],
    "thermogram_12.npy":
    [[0.060, 0.089, 0.000, 0.000, 0.000, 0.015, 0.000],
    [0.088, 0.060, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.127, 0.009, 0.000, 0.073, 0.000, 0.009, 0.000],
    [0.115, 0.083, 0.000, 0.000, 0.000, 0.122, 0.000],
    ],
    "thermogram_13.npy":
    [[0.086, 0.000, 0.000, 0.000, 0.000, 0.187, 0.420],
    [0.103, 0.000, 0.000, 0.000, 0.000, 0.049, 0.000],
    [0.119, 0.087, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.088, 0.063, 0.000, 0.000, 0.000, 0.030, 0.000],
    ],
    "thermogram_14.npy":
    [[0.167, 0.061, 0.000, 0.055, 0.000, 0.000, 0.000],
    [0.166, 0.083, 0.000, 0.076, 0.000, 0.000, 0.000],
    [0.131, 0.077, 0.000, 0.082, 0.000, 0.182, 0.000],
    [0.104, 0.093, 0.092, 0.051, 0.000, 0.360, 0.000],
    ],
    "thermogram_16.npy":
    [[0.000, 0.000, 0.000, 0.138, 0.192, 0.050, 0.000],
    [0.000, 0.000, 0.000, 0.069, 0.121, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.181, 0.206, 0.181, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.236, 0.388, 0.000],
    ],
    "thermogram_17.npy":
    [[0.000, 0.000, 0.000, 0.000, 0.100, 0.149, 0.943],
    [0.000, 0.000, 0.000, 0.000, 0.124, 0.146, 0.889],
    [0.000, 0.000, 0.000, 0.000, 0.149, 0.045, 0.210],
    [0.000, 0.000, 0.000, 0.000, 0.140, 0.050, 0.205],
    ],
    "thermogram_18.npy":
    [[0.000, 0.000, 0.077, 0.000, 0.000, 0.060, 0.359],
    [0.000, 0.000, 0.000, 0.000, 0.111, 0.091, 0.401],
    [0.000, 0.110, 0.063, 0.000, 0.063, 0.025, 0.000],
    [0.000, 0.000, 0.032, 0.173, 0.096, 0.073, 0.000],
    ],
    "thermogram_19.npy":
    [[0.000, 0.000, 0.000, 0.000, 0.050, 0.219, 2.101],
    [0.000, 0.000, 0.000, 0.000, 0.078, 0.000, 1.917],
    [0.000, 0.000, 0.000, 0.000, 0.037, 0.000, 1.940],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.196, 1.876],
    ],
    "thermogram_20.npy":
    [[0.000, 0.000, 0.000, 0.000, 0.275, 0.023, 0.494],
    [0.000, 0.000, 0.000, 0.000, 0.317, 0.046, 0.136],
    [0.000, 0.000, 0.000, 0.132, 0.382, 0.026, 0.000],
    [0.000, 0.000, 0.000, 0.086, 0.304, 0.100, 0.000],
    ],
    "thermogram_21.npy":
    [[0.000, 0.000, 0.000, 0.059, 0.122, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.084, 0.131, 0.046, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.224, 0.030, 0.000],
    [0.000, 0.000, 0.000, 0.063, 0.249, 0.000, 0.000]],
    "thermogram_22.npy":
    [[0.000, 0.096, 0.052, 0.000, 0.000, 0.176, 0.000],
    [0.000, 0.050, 0.000, 0.000, 0.127, 0.096, 0.000],
    [0.000, 0.000, 0.000, 0.084, 0.173, 0.000, 0.000],
    [0.000, 0.122, 0.000, 0.000, 0.094, 0.038, 0.000],
    ],
    "thermogram_23.npy":
    [[0.000, 0.176, 0.000, 0.000, 0.176, 0.119, 0.000],
    [0.000, 0.080, 0.000, 0.000, 0.200, 0.044, 0.000],
    [0.000, 0.118, 0.000, 0.000, 0.201, 0.078, 0.000],
    [0.000, 0.325, 0.000, 0.000, 0.259, 0.175, 0.000],
    ],
    "thermogram_24.npy":
    [[0.000, 0.156, 0.000, 0.000, 0.231, 0.140, 0.000],
    [0.000, 0.065, 0.000, 0.000, 0.236, 0.225, 0.000],
    [0.000, 0.072, 0.000, 0.034, 0.323, 0.146, 0.000],
    [0.000, 0.082, 0.000, 0.000, 0.238, 0.183, 0.000],
    ],
    "thermogram_25.npy":
    [[0.000, 0.103, 0.055, 0.000, 0.138, 0.218, 0.000],
    [0.000, 0.137, 0.000, 0.000, 0.119, 0.014, 0.000],
    [0.000, 0.163, 0.000, 0.000, 0.174, 0.000, 0.000],
    [0.000, 0.221, 0.000, 0.000, 0.103, 0.042, 0.000],
    ],
    "thermogram_26.npy":
    [[0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.660],
    [0.000, 0.000, 0.000, 0.000, 0.102, 0.000, 1.650],
    [0.000, 0.000, 0.480, 0.000, 0.000, 0.000, 1.926],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.844],
    ],
    "thermogram_27.npy":
    [[0.000, 0.000, 0.000, 0.000, 0.088, 0.365, 1.334],
    [0.000, 0.000, 0.000, 0.000, 0.122, 0.083, 0.582],
    [0.000, 0.000, 0.000, 0.000, 0.091, 0.097, 0.561],
    [0.000, 0.000, 0.000, 0.000, 0.072, 0.348, 0.750],
    ],
    "thermogram_30.npy":
    [[0.000, 0.103, 0.000, 0.000, 0.156, 0.176, 0.000],
    [0.000, 0.055, 0.000, 0.000, 0.119, 0.031, 0.000],
    [0.000, 0.125, 0.000, 0.000, 0.136, 0.125, 0.000],
    [0.000, 0.103, 0.000, 0.000, 0.109, 0.219, 0.000],
    ],
    "thermogram_31.npy":
    [[0.000, 0.076, 0.000, 0.000, 0.128, 0.161, 0.000],
    [0.000, 0.040, 0.000, 0.000, 0.118, 0.037, 0.000],
    [0.000, 0.076, 0.000, 0.000, 0.108, 0.055, 0.000],
    [0.000, 0.075, 0.000, 0.000, 0.055, 0.162, 0.000],
    ],
    "thermogram_32.npy":
    [[0.000, 0.075, 0.000, 0.000, 0.073, 0.148, 0.000],
    [0.000, 0.044, 0.000, 0.000, 0.109, 0.000, 0.000],
    [0.000, 0.200, 0.000, 0.000, 0.195, 0.117, 0.000],
    [0.000, 0.164, 0.000, 0.000, 0.163, 0.022, 0.000],
    ],
    "thermogram_33.npy":
    [[0.000, 0.063, 0.000, 0.081, 0.153, 0.218, 0.000],
    [0.000, 0.117, 0.000, 0.000, 0.189, 0.175, 0.000],
    [0.000, 0.088, 0.000, 0.000, 0.150, 0.066, 0.000],
    [0.000, 0.071, 0.000, 0.000, 0.147, 0.123, 0.000],
    ],
    "thermogram_34.npy":
    [[0.000, 0.061, 0.080, 0.000, 0.149, 0.222, 0.000],
    [0.000, 0.098, 0.000, 0.082, 0.000, 0.262, 0.000],
    [0.000, 0.106, 0.000, 0.094, 0.000, 0.250, 0.000],
    [0.000, 0.068, 0.000, 0.000, 0.000, 0.000, 0.000]
    ]
    }

    labels = {
    "thermogram_7.npy": 
    [0, 0, 1, 0],
    "thermogram_8.npy":
    [0, 0, 0, 0],
    "thermogram_9.npy":
    [0, 0, 0, 0],
    "thermogram_10.npy":
    [0, 0, 0, 0],
    "thermogram_11.npy":
    [0, 1, 0, 0],
    "thermogram_12.npy":
    [0, 0, 0, 0],
    "thermogram_13.npy":
    [1, 0, 0, 0],
    "thermogram_14.npy":
    [1, 1, 0, 1],
    "thermogram_16.npy":
    [0, 0, 0, 1],
    "thermogram_17.npy":
    [1, 1, 1, 1],
    "thermogram_18.npy":
    [1, 1, 0, 0],
    "thermogram_19.npy":
    [1, 1, 1, 1],
    "thermogram_20.npy":
    [1, 1, 1, 1],
    "thermogram_21.npy":
    [0, 0, 0, 0],
    "thermogram_22.npy":
    [0, 0, 0, 0],
    "thermogram_23.npy":
    [0, 0, 0, 1],
    "thermogram_24.npy":
    [0, 0, 1, 0],
    "thermogram_25.npy":
    [0, 0, 0, 0],
    "thermogram_26.npy":
    [1, 1, 1, 1],
    "thermogram_27.npy":
    [1, 1, 1, 1],
    "thermogram_30.npy":
    [0, 0, 0, 0],
    "thermogram_31.npy":
    [0, 0, 0, 0],
    "thermogram_32.npy":
    [0, 0, 0, 0],
    "thermogram_33.npy":
    [0, 0, 0, 0],
    "thermogram_34.npy":
    [0, 0, 0, 0],
    }

    names = ['velocity', 'size', 'temp', 'cooling_speed', 'appearance_rate', 'n_spatters', 'welding_zone_temp']
    new_names = ['total_spatters']
    if type == 'reduced':
        for name in names:
            new_names.append('mean_' + name)
            new_names.append('max_' + name)
            new_names.append('min_' + name)
    else:
        new_names.extend(names)

    defect_names = ['hu', 'hg', 'he','hp', 'hs', 'hm', 'hi']
    new_names.extend(defect_names)

    out = {key: {'A': [], 'B': [], 'C': [], 'D': []} for key in data.keys()}

    for key, value in data.items():  # iterate over thermograms
        for zone in out[key].keys():  # iterate over sections
            for metric in value.keys():  # iterate over feature in one thermogram
                if zone in metric:
                    out[key][zone].append(value[metric])  # add features values to thermogram: section

    # print(out.keys())
    # print(out['thermogram_21.npy'].keys())
    # f = np.array(out['thermogram_21.npy']['A'])
    # print(f.shape)
    # f = f.transpose(1, 2, 0)
    # print(f.shape)
    # print(f[1])

    if type == 'lstm':
        for key in out.keys():  
            for i, zone in enumerate(out[key].keys()):
                out[key][zone] = np.array(out[key][zone]).transpose(1, 2, 0)
                for h in quality[key][i]:
                    out[key][zone] = np.concatenate((out[key][zone], h * np.ones((*out[key][zone].shape[:2], 1))), axis=-1
                                                    )
    else:
        for key in out.keys():  
            for i, zone in enumerate(out[key].keys()):
                for h in quality[key][i]:
                    out[key][zone].append([h, ] * len(out[key][zone][0]))

    # print(out.keys())
    # print(out['thermogram_21.npy'].keys())
    # print(np.array(out['thermogram_21.npy']['A']))

    ds = []
    if type == 'lstm':
        for value in out.values():
            for d in value.values():
                ds.append(d)
        ds = np.concatenate(ds)
    else:
        for value in out.values():
            for d in value.values():
                for s in np.array(d).T.tolist():
                    ds.append(s)
        df = pd.DataFrame(ds, columns=new_names)

    if type == 'reduced':
        df = pd.DataFrame(ds, columns=new_names)

        df.hu = df.hu > quality_thresholds[0]
        df.hg = df.hg > quality_thresholds[1]
        df.he = df.he > quality_thresholds[2]
        df.hp = df.hp > quality_thresholds[3]
        df.hs = df.hs > quality_thresholds[4]
        df.hm = df.hm > quality_thresholds[5]
        df.hi = df.hi > quality_thresholds[6]

        df = df * 1

        is_defect = df[class_id]
        df = df.drop(defect_names, axis=1)
        df = df.drop([name for name in new_names if 'min' in name], axis=1)
        df = df.drop([name for name in new_names if 'max' in name], axis=1)

        return df, is_defect
    
    else:
        defect_id = defect_names.index(class_id)
        is_defect =  (ds[:, :, 8 + defect_id] > quality_thresholds[defect_id]) * 1
        ds = ds[:, :, :8]

        return ds, is_defect
