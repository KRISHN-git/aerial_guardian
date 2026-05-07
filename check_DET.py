import os
path = 'weights/finetune/drone_person_v1/weights/best.pt'
if os.path.exists(path):
    print(f'best.pt exists: {os.path.getsize(path)/1e6:.1f} MB')
else:
    print('NOT FOUND - training may still be running')
