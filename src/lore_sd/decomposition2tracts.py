import subprocess
import os
import argparse
import tqdm
import numpy as np

def regrid_VOI(voi, template, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    cmd = f'mrgrid {voi} regrid -template {template} {save_dir}/{voi.split("/")[-1]} -interp nearest -force -quiet'

    subprocess.run(cmd, shell=True)

def regrid_VOIs(voi_dir, template, save_dir):
    # For every directory in the VOI directory
    for voi in tqdm.tqdm(list(filter(lambda x: 'MNI' not in x and 'custom_' not in x and os.path.isdir(os.path.join(voi_dir, x)), os.listdir(voi_dir)))):
        for subdir in os.listdir(f'{voi_dir}/{voi}'):
            for f in os.listdir(f'{voi_dir}/{voi}/{subdir}'):
                # If the file is a nii.gz file
                if 'bin.nii.gz' in f:
                    # Apply the transformation to the file
                    regrid_VOI(f'{voi_dir}/{voi}/{subdir}/{f}', template, os.path.join(save_dir, voi, subdir))

    # For the custom VOIs, we only consider a few
    to_regrid_in_custom = ['cerebellum_Bil_X.nii.gz', 'cerebrum_hemi_LT_X_nv.nii.gz', 'cerebrum_hemi_RT_X_nv.nii.gz']
    for custom_voi in to_regrid_in_custom:
        regrid_VOI(f'{voi_dir}/custom_VOIs/{custom_voi}', template, os.path.join(save_dir, 'custom_VOIs'))

def generate_tract(voi_dir, custom_voi_dir, odfs, tracts, save_dir, tissue_seg, lesion=None):
    include_dirs = [f'{voi_dir}/{d}' for d in os.listdir(voi_dir) if 'incs' in d]
    exclude_dirs = [f'{voi_dir}/{d}' for d in os.listdir(voi_dir) if 'excs' in d]

    cerebellum_bundles = ['ICP', 'MCP', 'DRT']

    if 'custom_' in voi_dir:
        print(f'Custom tract {voi_dir} found. Skipping.')
        return

    # If there is any tck file in the voi_dir, this means that there are tracts within the ROI, so we regenerate them with more seeds

    # if any(os.path.exists(f'{voi_dir}/{tck_file}') for tck_file in tck_files):
    for odf in odfs:
        new_tck_file = f'{save_dir}/{odf.split("/")[-1].split(".")[0]}_tractography.tck'
        tckgen_cmd = get_tckgen_cmd(voi_dir, custom_voi_dir, save_dir, lesion, include_dirs, exclude_dirs, new_tck_file, odf, tissue_seg)

        subprocess.run(tckgen_cmd, shell=True)
        subprocess.run(f'chmod 774  {new_tck_file}', shell=True)

        size = int(subprocess.run(['tckinfo', new_tck_file, '-count'], stdout=subprocess.PIPE, text=True).stdout.strip().split()[-1])
        print(size)
        if size == 0:
            rm_cmd = f'rm {new_tck_file}'
            subprocess.run(rm_cmd, shell=True)

        # if 'custom_' in tract_name:
        #     print(f'Custom tract {tract_name} found. Skipping.')
        #     continue

        # # tckgen_cmd = get_tckedit_cmd(voi_dir, custom_voi_dir, save_dir, lesion, include_dirs, exclude_dirs, tract, tract_name)

        # # print(f'Running command: {tckgen_cmd}')

        # # subprocess.run(tckgen_cmd, shell=True)
        # # subprocess.run(f'chmod 774  {save_dir}/{tract_name}.tck', shell=True)

        # if not os.path.exists(f'{save_dir}/{tract_name}.tck'):
        #     print(f'Tract file {save_dir}/{tract_name}.tck does not exist. Skipping size check.')
        #     continue

        # size = int(subprocess.run(['tckinfo', f'{save_dir}/{tract_name}.tck', '-count'], stdout=subprocess.PIPE, text=True).stdout.strip().split()[-1])
        # if size == 0:
        #     rm_cmd = f'rm {save_dir}/{tract_name}.tck'
        #     subprocess.run(rm_cmd, shell=True)
        # else:
        #     print(f'Generated tract {tract_name} with size {size}. Regenerating tract...')
        #     tckgen_cmd = get_tckgen_cmd(voi_dir, custom_voi_dir, save_dir, lesion, include_dirs, exclude_dirs, tract_name, corresponding_odf)
        #     print(f'Running command: {tckgen_cmd}')
        #     subprocess.run(tckgen_cmd, shell=True)
        #     subprocess.run(f'chmod 774  {save_dir}/{tract_name}.tck', shell=True)

def get_tckedit_cmd(voi_dir, custom_voi_dir, save_dir, lesion, include_dirs, exclude_dirs, tract, tract_name):
    tckgen_cmd = f'tckedit {tract} {save_dir}/{tract_name}.tck ' + \
                f'-nthreads 100 -force '
        # If none of the 'cerebellum_bundles' is in the name of the VOI directory, exclude the cerebellum
        # if not any([cb in voi_dir for cb in cerebellum_bundles]):
        #     tckgen_cmd += f'-exclude {custom_voi_dir}/cerebellum_Bil_X.nii.gz '
    if 'LT' in voi_dir:
        tckgen_cmd += f'-exclude {custom_voi_dir}/cerebrum_hemi_RT_X_nv.nii.gz '
    elif 'RT' in voi_dir:
        tckgen_cmd += f'-exclude {custom_voi_dir}/cerebrum_hemi_LT_X_nv.nii.gz '

    for exc in exclude_dirs:
        f = os.listdir(exc)[0]
        tckgen_cmd += f'-exclude {exc}/{f} '

    for inc in include_dirs:
        f = os.listdir(inc)[0]
        tckgen_cmd += f'-include {inc}/{f} '

    if lesion is not None:
        tckgen_cmd += f'-include {lesion} '
    return tckgen_cmd

def get_tckgen_cmd(voi_dir, custom_voi_dir, save_dir, lesion, include_dirs, exclude_dirs, tract_name, odf, tissue_seg):
    tckgen_cmd = f'tckgen {odf} {tract_name} ' + \
                f'-nthreads 100 -force -select 10k -seeds 2M -quiet '
        # If none of the 'cerebellum_bundles' is in the name of the VOI directory, exclude the cerebellum
        # if not any([cb in voi_dir for cb in cerebellum_bundles]):
        #     tckgen_cmd += f'-exclude {custom_voi_dir}/cerebellum_Bil_X.nii.gz '

    data_dir = os.path.dirname(os.path.dirname(odf))

    if 'LT' in voi_dir:
        tckgen_cmd += f'-exclude {custom_voi_dir}/cerebrum_hemi_RT_X_nv.nii.gz '
    elif 'RT' in voi_dir:
        tckgen_cmd += f'-exclude {custom_voi_dir}/cerebrum_hemi_LT_X_nv.nii.gz '

    if 'lore_iax' in tract_name:
        tckgen_cmd += f'-cutoff .05 '
    elif 'lore_rfa' in tract_name:
        tckgen_cmd += f'-cutoff .05 '
    elif 'wm' in tract_name:
        tckgen_cmd += f'-cutoff .05 '

    for exc in exclude_dirs:
        f = os.listdir(exc)[0]
        tckgen_cmd += f'-exclude {exc}/{f} '

    for inc in include_dirs:
        f = os.listdir(inc)[0]
        tckgen_cmd += f'-include {inc}/{f} '
        tckgen_cmd += f'-seed_image {inc}/{f} '

    if lesion is not None:
        tckgen_cmd += f'-include {lesion} '
        # tckgen_cmd += f'-seed_image {lesion} '

    tckgen_cmd += f'-act {data_dir}/5tt/5tt_lesion_separate_dwi.mif -backtrack'
    return tckgen_cmd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment tracts')
    parser.add_argument('voi_dir', type=str, help='Path to the directory containing the VOIs')
    parser.add_argument('odfs', type=str, help='List of odf files separated by commas')
    parser.add_argument('tracts', type=str, help='List of tract files separated by commas')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    parser.add_argument('template', type=str, help='Path to the template file for regridding')
    parser.add_argument('--lesion', type=str, help='Path to the lesion mask', default=None)

    args = parser.parse_args()

    voi_dir = args.voi_dir
    # Find all subdirs that to no have MNI in their name
    voi_dirs = [f'{voi_dir}/{d}' for d in os.listdir(voi_dir) if 'MNI' not in d and 'done' not in d and 'custom' not in d]

    tracts = args.tracts.split(',')
    tracts = [os.path.join(os.getcwd(), o) for o in tracts]

    odfs = args.odfs.split(',')
    odfs = [os.path.join(os.getcwd(), o) for o in odfs]

    os.makedirs(args.output_dir, exist_ok=True)

    lesion = args.lesion if args.lesion is not None else None

    regrid_VOIs(voi_dir, f'{args.template}', args.output_dir)

    vois = list(filter(lambda x: os.path.isdir(os.path.join(args.output_dir, x)), os.listdir(args.output_dir)))
    for voi in tqdm.tqdm(vois):
            print(f'Generating tract {voi}')
            generate_tract(f'{args.output_dir}/{voi}', f'{args.output_dir}/custom_VOIs', odfs,
                        tracts, os.path.join(args.output_dir, voi), os.path.join(os.path.dirname(voi_dir), '5tt/5tt_lesion_separate_dwi.mif'), lesion)
            print(f'Finished generating tract {voi}')
