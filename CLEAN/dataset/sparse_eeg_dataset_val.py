import torch
from torch.utils.data import Dataset
import numpy as np
import mne
from torch import from_numpy as np2TT
import random

class SparseEEGDataset_val(Dataset):
    def __init__(self, 
                 x, 
                 y, 
                 query, 
                 channel_order, 
                 cfg_dataset, 
                 train=True, 
                 num_feat=1, 
                 use_montage='tuh', 
                 montage='standard_1020', 
                 io_same = False, 
                 ):
        """
        Args:
            x (torch.Tensor): Input EEG data of shape [number of EEG segments, channels, sequence length].
            y (torch.Tensor): Target EEG data of shape [number of EEG segments, channels, sequence length].
            num_inputs (int): Number of input points to sample.
            num_outputs (int): Number of output points to sample.
            train (bool): If True, random sampling is applied; otherwise, deterministic sampling.
        """
        super(SparseEEGDataset_val, self).__init__()
        assert len(x) == len(y), "Input data and target data must have the same number of segments."
        self.x = x
        self.y = y
        self.query = query
        self.train = train
        self.channels = x.shape[1]
        self.sequence_length = x.shape[2]
        self.channel2pos_dict = mne.channels.make_standard_montage(montage).get_positions()['ch_pos']
        stacked_positions = np.vstack(list(self.channel2pos_dict.values()))
        # Calculate min and max
        min_pos = stacked_positions.min()
        max_pos = stacked_positions.max()
        # Rescale each array in the dictionary
        for key in self.channel2pos_dict:
            self.channel2pos_dict[key] = 100 * (self.channel2pos_dict[key] - min_pos) / (max_pos - min_pos) #rescaled to [0, 100]  
        self.channel2pos_dict = {k.lower().replace('eeg', '').strip(): v for k, v in self.channel2pos_dict.items()}
        self.channel_order = channel_order
        self.channel_pos = np.array([self.ch2pos(ch) for ch in self.channel_order]) #3d electrode positions[channels, 3]
        self.sfreq = cfg_dataset["sfreq"]
        self.channel_index_map = {name: idx for idx, name in enumerate(channel_order)}
        self.use_montage = use_montage
        self.num_feat = num_feat
        self.io_same = io_same
        self.tuh_montage_pairs = [
            ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
            ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
            ('A1', 'T3'), ('T3', 'C3'), ('C3', 'CZ'), ('CZ', 'C4'),
            ('C4', 'T4'), ('T4', 'A2'), ('FP1', 'F3'), ('F3', 'C3'),
            ('C3', 'P3'), ('P3', 'O1'), ('FP2', 'F4'), ('F4', 'C4'),
            ('C4', 'P4'), ('P4', 'O2')
            ]

        # Positional encoding: Generate positions for channels and time
        # Channel positions: [channels, 1]
        #self.channel_pos = torch.arange(self.channels).float().unsqueeze(1)
        # Time positions: [1, sequence_length]
        self.time_pos = torch.arange(start=0, end=self.sequence_length/cfg_dataset['sfreq'], step=1/cfg_dataset['sfreq']).float().unsqueeze(0)*173 #rescale: 1 second equals 173

    def ch2pos(self, channel: str) -> torch.FloatTensor:
        """
        fault tolerant channel to position lookup -- ignores EEG prefix and is not case sensitive
        returns [0, 0, 0] if channel is not found
        """
        # if relative channel:
        if '-' in channel: # relative channel; typically relative to ref --> TODO
            channels = channel.split('-')
            pos1 = self.ch2pos(channels[0])
            pos2 = self.ch2pos(channels[1])
            return (pos1 + pos2) / 2
        channel = channel.lower().replace('eeg', '').strip()
        # throw warning if channel is not found
        if channel not in self.channel2pos_dict:
            print(f"Channel {channel} not found in channel2pos_dict")
        return self.channel2pos_dict.get(channel, [0, 0, 0])

    def generate_random_pairs(self, channel_nr, N):
        """
        Generate N random pairs of channel indices without repetition.
        """
        all_pairs = [(i, j) for i in range(channel_nr) for j in range(channel_nr) if i != j]
        np.random.shuffle(all_pairs)
        return all_pairs[:N]    

    def create_montage(self, eeg_segment, montage = 'tuh', montage_pairs=None, debug = False):
        bipolar_data = []
        bipolar_names = []
        bipolar_positions = []
        if montage == 'tuh':
            #TODO use self.channel_order
            montage_pairs = self.tuh_montage_pairs

            # Create the bipolar pairs
            for anode, cathode in montage_pairs:
                try:
                    anode_idx = self.channel_index_map[anode]
                    cathode_idx = self.channel_index_map[cathode]
                    bipolar_signal = eeg_segment[anode_idx] - eeg_segment[cathode_idx]
                    bipolar_data.append(bipolar_signal)
                    bipolar_names.append(f"{anode}-{cathode}")
                    # Combine 3D positions of anode and cathode into a (6,) array
                    combined_positions = np.concatenate([self.channel_pos[anode_idx], self.channel_pos[cathode_idx]])
                    bipolar_positions.append(combined_positions)
                except KeyError:
                  if debug:
                      print(f"Skipping pair {anode}-{cathode}: One or both channels are missing.")

            # Convert the bipolar data list to a numpy array
            bipolar_data = np.array(bipolar_data)
            bipolar_positions = np.array(bipolar_positions)  # Shape: (number_of_pairs, 6)

            if debug:
                # Output
                print("Bipolar Data Shape:", bipolar_data.shape)
                print("Bipolar Positions Shape:", bipolar_positions.shape)
                print("Bipolar Channels:", bipolar_names)
                print("Bipolar Data", bipolar_data)
                print("EEG segment", eeg_segment)

        elif montage == 'tuh_rand':
            # Create the bipolar pairs
            for anode, cathode in montage_pairs:
                try:
                    anode_idx = self.channel_index_map[anode]
                    cathode_idx = self.channel_index_map[cathode]
                    bipolar_signal = eeg_segment[anode_idx] - eeg_segment[cathode_idx]
                    bipolar_data.append(bipolar_signal)
                    bipolar_names.append(f"{anode}-{cathode}")
                    # Combine 3D positions of anode and cathode into a (6,) array
                    combined_positions = np.concatenate([self.channel_pos[anode_idx], self.channel_pos[cathode_idx]])
                    bipolar_positions.append(combined_positions)
                except KeyError:
                  if debug:
                      print(f"Skipping pair {anode}-{cathode}: One or both channels are missing.")

            # Convert the bipolar data list to a numpy array
            bipolar_data = np.array(bipolar_data)
            bipolar_positions = np.array(bipolar_positions)  # Shape: (number_of_pairs, 6)
        elif montage == 'random':
            # Create the bipolar pairs
            for anode_idx, cathode_idx in montage_pairs:
                try:
                    bipolar_signal = eeg_segment[anode_idx] - eeg_segment[cathode_idx]
                    bipolar_data.append(bipolar_signal)
                    bipolar_names.append(f"{self.channel_order[anode_idx]}-{self.channel_order[cathode_idx]}")
                    # Combine 3D positions of anode and cathode into a (6,) array
                    combined_positions = np.concatenate([self.channel_pos[anode_idx], self.channel_pos[cathode_idx]])
                    bipolar_positions.append(combined_positions)
                except KeyError:
                  if debug:
                      print(f"Skipping pair {anode}-{cathode}: One or both channels are missing.")

            # Convert the bipolar data list to a numpy array
            bipolar_data = np.array(bipolar_data)
            bipolar_positions = np.array(bipolar_positions)  # Shape: (number_of_pairs, 6)
            
        elif montage == 'user_specific':
            montage_pairs = self.query['query_montage_pairs']
            # Create the bipolar pairs
            for anode, cathode in montage_pairs:
                try:
                    anode_idx = self.channel_index_map[anode]
                    cathode_idx = self.channel_index_map[cathode]
                    bipolar_signal = eeg_segment[anode_idx] - eeg_segment[cathode_idx]
                    bipolar_data.append(bipolar_signal)
                    bipolar_names.append(f"{anode}-{cathode}")
                    # Combine 3D positions of anode and cathode into a (6,) array
                    combined_positions = np.concatenate([self.channel_pos[anode_idx], self.channel_pos[cathode_idx]])
                    bipolar_positions.append(combined_positions)
                except KeyError:
                  if debug:
                      print(f"Skipping pair {anode}-{cathode}: One or both channels are missing.")

            # Convert the bipolar data list to a numpy array
            bipolar_data = np.array(bipolar_data)
            bipolar_positions = np.array(bipolar_positions)  # Shape: (number_of_pairs, 6)

            if debug:
                # Output
                print("Bipolar Data Shape:", bipolar_data.shape)
                print("Bipolar Positions Shape:", bipolar_positions.shape)
                print("Bipolar Channels:", bipolar_names)
                print("Bipolar Data", bipolar_data)
                print("EEG segment", eeg_segment)
            
        elif montage == None or 'no_montage':
            bipolar_data = eeg_segment
            bipolar_positions =  self.channel_pos         # Shape: (number_of_pairs, 3)

        return bipolar_data, bipolar_positions

    def normalize_segments(self, eeg_data_x, eeg_data_y, debug=False):
        # Calculate the 95th percentile of the absolute value for each channel in the segment
        percentiles = np.percentile(np.abs(eeg_data_x), 95, axis=-1, keepdims=True)

        if debug:
            # Check for zero values in the percentiles
            if np.any(percentiles == 0):
                print("Zero value detected in percentiles:")
                print("Percentiles:", percentiles)
                print("EEG Data (eeg_data_x):", eeg_data_x)
                print("EEG Data (eeg_data_y):", eeg_data_y)

        percentiles = np.where(percentiles == 0, 1e-7, percentiles) #avoid zero values by replacing with 1e-7

        # Normalize each EEG segment by dividing by the 95th percentile of the absolute value of each channel
        return eeg_data_x / percentiles, eeg_data_y / percentiles, percentiles


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Retrieve the EEG segment and ground truth
        x_segment = self.x[idx]  # Shape: [channels, sequence_length]
        y_segment = self.y[idx]  # Shape: [channels, sequence_length]

        montage_pairs = self.generate_random_pairs(len(self.channel_order), N=20) if self.use_montage == 'random' else None
        if self.use_montage == "tuh_rand":
            montage_pairs = self.tuh_montage_pairs
            random.shuffle(montage_pairs)

        x_segment, ch_positions_6d = self.create_montage(x_segment, montage=self.use_montage, montage_pairs = montage_pairs) #montaged data
        y_segment, _ = self.create_montage(y_segment, montage=self.use_montage, montage_pairs = montage_pairs)   #montaged data
        x_segment, y_segment, percentiles = self.normalize_segments(x_segment, y_segment)

        x_segment = np2TT(x_segment)
        y_segment = np2TT(y_segment)

        ch_number_input = x_segment.shape[0]
        #ch_number_query = len(self.query['query_montage_pairs'])
        ch_number_query = ch_number_input #TODO

        #features_x = torch.stack((x_segment, v_x, a_x), dim=-1)   #shape (num_inputs, 3)
        #features_y = torch.stack((y_segment, v_y, a_y), dim=-1)   #shape (num_outputs, 3)  #TODO

        #TODO distinguish between self.num_feat = 1 (only amplitude) or 2 (amplitude, velocity) or 3(ampl, vel, acceleration), needs to be taken care of in encoder and decoder as well!

        features_x = torch.stack((x_segment,), dim=-1)   #shape (num_inputs, num_features)
        features_y = torch.stack((y_segment,), dim=-1)   #shape (num_outputs, num_features)  #TODO

        # Create positional encodings for channels and time
        pos_channels = torch.tensor(np.repeat(ch_positions_6d[:, np.newaxis, :], self.sequence_length, axis=1), dtype=torch.float32) # Shape: [channels, sequence_length, 6]
        pos_channels = pos_channels.permute(2,0,1) # Shape: [6, channels, sequence_length]
        #pos_channels = pos_channels/(pos_channels.max())*100   # reshape to [0,100]

        pos_time = self.time_pos.repeat(ch_number_input, 1).unsqueeze(2)  # Shape: [channels, sequence_length, 1]
        pos_time = pos_time.permute(2,0,1) # Shape: [1, channels, sequence_length]
        #pos_time = pos_time/(pos_time.max())*173 # reshape to [0,173] 173 is maximum Euclidean distance with positions

        # Combine positions into a 3D tensor
        pos_encoding = torch.cat((pos_channels, pos_time), dim=0) # Shape: [7, channels, sequence_length]

        # Subsample random input points
        input_feat = features_x.reshape(-1, features_x.shape[-1])
        input_pos = pos_encoding.reshape(7, -1)
        target_feat = features_y.reshape(-1,features_y.shape[-1])
        target_pos = pos_encoding.reshape(7, -1)

        #### query ####
        query_time =  torch.arange(start=0, end=self.sequence_length/self.sfreq, step=1/self.query['query_freq']).float().unsqueeze(0)*173
        query_time =  query_time.repeat(ch_number_query, 1).unsqueeze(2)  # Shape: [channels, sequence_length, 1]
        query_time = query_time.permute(2,0,1) # Shape: [1, channels, sequence_length]
        #query_time = query_time/(query_time.max())*173 # reshape to [0,173] 173 is maximum Euclidean distance with positions
        
        query_spatial_pos = torch.tensor(np.repeat(ch_positions_6d[:, np.newaxis, :], self.sequence_length*self.query['query_freq']/self.sfreq, axis=1), dtype=torch.float32) # Shape: [channels, sequence_length, 6]
        query_spatial_pos = query_spatial_pos.permute(2,0,1) # Shape: [6, channels, sequence_length stretched according to query freq]
        #query_spatial_pos = query_spatial_pos/(query_spatial_pos.max())*100   # reshape to [0,100]

        query_pos_encoding = torch.cat((query_spatial_pos, query_time), dim=0)# Shape: [7, channels, sequence_length stretched according to query freq]
        query_pos_encoding = query_pos_encoding.reshape(7, -1) 
        
        return dict(
            index=idx,
            input_feat=input_feat.to(torch.float32),
            input_pos=input_pos.permute(1,0).to(torch.float32),
            target_feat=target_feat.to(torch.float32),
            target_pos=target_pos.permute(1,0).to(torch.float32),
            query_pos=query_pos_encoding.permute(1,0).to(torch.float32),
            norm_factor = percentiles,
        )