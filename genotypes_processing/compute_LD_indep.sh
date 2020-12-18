/home/hv270/Plink2/plink2 --bfile /home/hv270/rds/rds-who1000-cbrc/user/wja24/shared/hv270/interval_genotypes/chr_$1 \
	--maf 0.01 \
	--indep-pairwise 50kb 0.01 \
	--out /home/hv270/rds/rds-who1000-cbrc/user/wja24/shared/hippolyte_only/interval_processing/indep_snps_sets/chr_$1
