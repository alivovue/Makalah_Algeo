# Makalah_Algeo
Untuk run training : 

Landing : 
python train_landing.py --csv landing.csv --out model_landing.json --lambda_ 0.01 --min_team_samples 1

Rotation : 
python train_rotation.py --csv rotation.csv --out model_rotation.json --lambda_ 0.01 --min_team_pairs 1