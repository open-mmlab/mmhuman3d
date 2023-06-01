import os 
import ftplib
import sys
import tqdm
from multiprocessing import Pool


avaliable_servers = ['1988']

ip_dict = {'1988': '10.198.8.138'}

def request_files(files, server_path, local_path, server_name='1988'):
    assert server_name in avaliable_servers, f'Unsupported server name: {server_name}'

    session = ftplib.FTP(ip_dict[server_name], user=user, passwd=pwd)
    session.cwd(server_path)

    names = session.nlst()

    for file in tqdm.tqdm(files, desc='Downloading files',
                          position=0, leave=False):
        if os.path.sep in file:
            os.makedirs(os.path.join(local_path, os.path.dirname(file)), exist_ok=True)
        session.retrbinary(cmd="RETR " + file, 
                       callback=open(os.path.join(local_path, file), "wb").write)
        
        
def request_files_name(fname, server_path, local_path, server_name='1988', exclude=''):
    assert server_name in avaliable_servers, f'Unsupported server name: {server_name}'

    session = ftplib.FTP(ip_dict[server_name], user=user, passwd=pwd)
    session.cwd(server_path)

    names = session.nlst()

    files = [f for f in names if fname in f]
    if exclude != '':
        files = [f for f in files if exclude not in f]
    for file in tqdm.tqdm(files, desc='Downloading files',
                          position=1, leave=False):
        session.retrbinary(cmd="RETR " + file, 
                    callback=open(os.path.join(local_path, file), "wb").write)
    return files


def request_folders(folders, server_name):

    return
    assert server_name in avaliable_servers, f'Unsupported server name: {server_name}'


    session = ftplib.FTP(ip_dict[server_name], user=user, passwd=pwd)
    session.cwd(path_name)
    names = session.nlst()


def downloadFiles(session, base, filename, destination):
    try:
        path = base + '/' + filename
        session.cwd(path)
        # clone path to destination
        os.chdir(destination)
        os.mkdir(os.path.join(destination, filename))
        print(os.path.join(destination, filename) + " built")
    except OSError:
        # folder already exists at destination
        pass
    except ftplib.error_perm:
        print("error: could not change to " + path)
        sys.exit("ending session")

    # list children:
    filelist = session.nlst()

    for file in filelist:
        try:
            # this will check if file is folder:
            session.cwd(path + '/' + file)
            # if so, explore it:
            downloadFiles(session, base=base, filename=filename + '/' + file, destination=destination)
        except ftplib.error_perm:
            # not a folder with accessible content
            # download & return
            os.chdir(os.path.join(destination, filename))
            # possibly need a permission exception catch:
            session.retrbinary(cmd="RETR " + file, callback=open(os.path.join(destination, filename, file), "wb").write)
    return

if __name__ == '__main__':

    humandata_datasets = [ # name, glob pattern, exclude pattern
        ('arctic', 'p1_train.npz', ''),
        ('bedlam', 'bedlam_train.npz', ''),
        ('behave', 'behave_train_230516_231_downsampled.npz', ''),
        ('chi3d', 'CHI3D_train_230511_1492_*.npz', ''),
        ('crowdpose', 'crowdpose_neural_annot_train_new.npz', ''),
        ('lspet', 'eft_lspet.npz', ''),
        ('ochuman', 'eft_ochuman.npz', ''),
        ('posetrack', 'eft_posetrack.npz', ''),
        ('egobody_ego', 'egobody_egocentric_train_230425_065_fix_betas.npz', ''),
        ('egobody_kinect', 'egobody_kinect_train_230503_065_fix_betas.npz', ''),
        ('fit3d', 'FIT3D_train_230511_1504_*.npz', ''),
        ('gta', 'gta_human2multiple_230406_04000_0.npz', ''),
        ('ehf', 'h4w_ehf_val_updated_v2.npz', ''),  # use humandata ehf to get bbox
        ('humansc3d', 'HumanSC3D_train_230511_2752_*.npz', ''),
        ('instavariety', 'insta_variety_neural_annot_train.npz', ''),
        ('mpi_inf_3dhp', 'mpi_inf_3dhp_neural_annot_train.npz', ''),
        ('mtp', 'mtp_smplx_train.npz', ''),
        ('muco3dhp', 'muco3dhp_train.npz', ''),
        ('prox', 'prox_train_smplx_new.npz', ''),
        ('renbody', 'renbody_train_230525_399_*.npz', ''),
        ('renbody_highres', 'renbody_train_highrescam_230517_399_*_fix_betas.npz', ''),
        ('rich', 'rich_train_fix_betas.npz', ''),
        ('spec', 'spec_train_smpl.npz', ''),
        ('ssp3d', 'ssp3d_230525_311.npz', ''),
        ('synbody_magic1', 'synbody_amass_230328_02172.npz', ''),
        ('synbody', 'synbody_train_230521_04000_fix_betas.npz', ''),
        ('talkshow', 'talkshow_smplx_*.npz', 'path'),
        ('up3d', 'up3d_trainval.npz', ''),
    ]

    num_proc = 4
    with Pool(num_proc) as p:
        r = list(tqdm.tqdm(p.imap(request_files, humandata_datasets), 
                      total=len(humandata_datasets), desc='sequences'))

    # request_files_name(fname='p1_train.npz', 
    #               server_path='/lustre/share_data/weichen1/all_humandata', 
    #               local_path='/mnt/d/annotations_1988', 
    #               server_name='1988')