% Main script to call data processing function of TUSZ data
% Dataset available at https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml

startsub = 1;
stopsub = 200;
folder = 'train';
root = "" + folder; %TODO!
montage =  "03_tcp_ar_a"; %"01_tcp_ar"; %"03_tcp_ar_a"; %01_tcp_ar; % ;
saveroot = 'dataset_processed/' + montage + '/' + folder; %TODO!
chanLocsPath = '/utils/Standard-10-20-Cap81-EDIT.ced';
%chanLocsPath = '/Standard-10-20-Cap81-EDIT.ced';
channels = {'FP1', 'FP2', 'F3', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ', 'A1', 'A2'};
removeSeiz = false;
ica = true;
saveArrays = false;
saveLabels = true;
normalize = false;
resampling_freq = 250;
car_reref = true;

% call main function
process_TUH_data2(startsub,stopsub,montage,root,saveroot, chanLocsPath, channels, removeSeiz, ica, saveArrays, saveLabels, normalize, resampling_freq, car_reref)

               