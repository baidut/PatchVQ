from . import *
from .single_vid2mos import *
from tqdm import tqdm
from .video_utils import video_process
# from ....util_scripts.generate_video_jpgs import video_process

class TestVideos(SingleVideo2MOS):

    def get_df(self):
        # always generate a new one? - no, unless it doesn't exist
        csv_file = self.path/self.csv_labels
        if not csv_file.exists():
            print('generating jpg files:')
            files = list((self.path/'mp4').glob('*.mp4'))
            print(f'Found {len(files)} files.')
            files = [f for f in files if not (self.path/self.folder/(f.stem)).exists()]
            print(f'Generating {len(files)} files.')
            for f in tqdm(files):
                print(f'Extracting frames: {f}')
                video_process(video_file_path=f,
                              dst_root_path=self.path/self.folder,
                              ext='.mp4', fps=-1, size=-1)
            # print(f'Skip cuz exists: {jpg_folder}')

            print('prepare label csv files')
            df = pd.DataFrame({self.fn_col: [f.stem for f in files], self.label_col: -1}) # np.nan
            df.to_csv(csv_file, index=False)
        return super().get_df()
