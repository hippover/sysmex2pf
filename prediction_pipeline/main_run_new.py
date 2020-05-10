import sys

sys.path.append("/home/hv270/platelets/")
print("Appended")
print(sys.path)

from preprocessing import *
from select_ids import *
from FACS_phenotypes import *
from build_features import *
from training_new import *
import argparse
import time

export_path = "/home/hv270/rds/rds-who1000-cbrc/user/wja24/shared/hv270/runs/"

if __name__ == "__main__":

    #timestr = time.strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="no_name")
    parser.add_argument("--hor" , type=int, help="# of steps to cut the PC1 into (default 40)", default=40)
    parser.add_argument("--vert", type=int, help="# of steps to cut the PC2 into (default 40)", default=40)
    parser.add_argument('--adj', dest='adjust', action='store_true')
    parser.add_argument('--no-adj', dest='adjust', action='store_false')
    parser.set_defaults(feature=True)

    args = parser.parse_args()
    tag = args.tag
    hor_steps = args.hor
    vert_steps = args.vert
    adjust = args.adjust

    tag += "-%d-%d" % (hor_steps, vert_steps)

    out_folder = export_path + tag
    print("Running %s" % out_folder)

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    ### Cluster Sysmex ###
    print("Cluster Sysmex")

    fc_pheno = load_fc_pheno()
    if not os.path.exists("%s/sample_weights.hdf" % out_folder):
        print("Preprocessing")
        sys_phen = load_sys_phenotypes()

        sys_df_init = load_Sysmex()

        sys_df_full = sys_df_init.copy()
        
        sys_df = tag_platelets(sys_df_full)

        f = "%s/sys_df.hdf" % out_folder #% datetime.utcnow().strftime("%d%m%Y_%H%M")
        print("Saving to %s " % f)
        sys_df.to_hdf(f,key="sys_df")

        ### Select IDs

        print("Select IDs")

        sys_df = pd.read_hdf("%s/sys_df.hdf" % out_folder ,key="sys_df")
        df_fc = pd.read_hdf("%s/df_fc.hdf" % out_folder, key="df_fc")

        sys_df,df_fc,fc_pheno,sys_phen,good_ids = get_intersection_dataframes(sys_df,df_fc,fc_pheno,sys_phen,only_wb=True)

        with open("%s/good_ids.txt" % out_folder, 'w+') as f:
            for item in good_ids:
                f.write("%s\n" % item)

        sys_df.to_hdf("%s/sys_df.hdf" % out_folder ,key="sys_df")
        df_fc.to_hdf("%s/df_fc.hdf" % out_folder, key="df_fc")
        fc_pheno.to_hdf("%s/fc_pheno.hdf" % out_folder, key="fc_pheno")
        sys_phen.to_hdf("%s/sys_phen.hdf" % out_folder, key="sys_phen")

        ### Phenotypes

        print("Build phenotypes")

        df_fc = pd.read_hdf("%s/df_fc.hdf" % out_folder, key="df_fc")

        fc_phenotypes, sample_weights = get_activations_and_weights(df_fc,n_quantiles=5,plot=False)

        fc_phenotypes.to_hdf("%s/fc_phenotypes.hdf" % out_folder,key="fc_phenotypes")
        sample_weights.to_hdf("%s/sample_weights.hdf" % out_folder,key="sample_weights")

    ### Training
    else:
        print("Skipping preprocessing")
    print("Training...")

    with open("%s/good_ids.txt" % out_folder) as f:
        good_ids = f.readlines()
        good_ids = [g[:-1] for g in good_ids]

    sys_df = pd.read_hdf("%s/sys_df.hdf" % out_folder,key="sys_df")
    sys_phen = pd.read_hdf("%s/sys_phen.hdf" % out_folder,key="sys_phen")
    sample_weights = pd.read_hdf("%s/sample_weights.hdf" % out_folder,key="sample_weights")
    all_fc_pheno = pd.read_hdf("%s/fc_phenotypes.hdf" % out_folder,key="fc_phenotypes")

    print("... to export")

    train_and_export(sys_df,sys_phen,all_fc_pheno,sample_weights,good_ids,hor_steps,vert_steps,out_folder,adjust=adjust)

    print("Done")
