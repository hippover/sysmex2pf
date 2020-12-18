/home/hv270/Plink2/plink2 --bgen /home/hv270/rds/rds-who1000-cbrc/user/wja24/shared/alexander_only/impute_$1_interval.bgen \
	--sample /home/hv270/rds/rds-who1000-cbrc/user/wja24/shared/alexander_only/interval.sample \
	--threads 8 \
	--maf 0.01 \
	--make-bed --out /home/hv270/rds/rds-who1000-cbrc/user/wja24/shared/hv270/interval_genotypes/chr_$1
