/home/hv270/Plink2/plink2 --bfile /home/hv270/rds/rds-who1000-cbrc/user/wja24/shared/hv270/interval_genotypes/chr_1 \
	--extract /home/hv270/rds/rds-who1000-cbrc/user/wja24/shared/hippolyte_only/interval_processing/indep_snps_sets/chr_1.prune.in \
	--keep /home/hv270/rds/rds-who1000-cbrc/user/wja24/shared/hippolyte_only/interval_processing/king-cutoff.king.cutoff.in.id \
	--maf 0.01 \
	--threads 24 \
	--pca 10 approx \
	--out /home/hv270/rds/rds-who1000-cbrc/user/wja24/shared/hippolyte_only/interval_processing/principal_components/PC 
