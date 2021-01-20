import numpy as np
import pandas as pd
import glob

def loadData(data_path=r"data/"):
    gt_data = pd.read_csv(data_path + 'gt_data.txt', names=['X', 'Y', 'Orientation'], sep=' ')
    map_data = pd.read_csv(data_path + 'map_data.txt', names=['X', 'Y', '# landmark'])
    control_data = pd.read_csv(data_path + 'control_data.txt', names=['velocity', 'Yaw rate'], sep=' ')

    # observation = pd.read_csv('data/observation/observations_000001.txt', names = ['X cord','Y cord'], sep=' ')

    result = [(x, y, landmark) for x, y, landmark in zip(map_data['X'], map_data['Y'], map_data['# landmark'])]
    landarkList = []
    for res in result:
        # l = pUtils.LandMark(res[0], res[1], res[2])
        landarkList.append((res[0], res[1], res[2]))
    landarkList = np.array(landarkList)

    obs_path = glob.glob(data_path + r"/observation/observations*.txt")
    obs_path.sort()

    print('Loading the Observations..')
    observation = []
    for file_path in obs_path:
        observationTmp = pd.read_csv(file_path, names=['X cord', 'Y cord'], sep=' ')
        observation.append(observationTmp)
    print('Loading Done!')

    return observation, control_data, gt_data, landarkList