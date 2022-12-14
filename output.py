import pandas as pd
from pyntcloud import PyntCloud

def save (output_points, output_colors):
	output_file = 'point_cloud.ply'
	d = {
		'x': 			output_points[:,0],
		'y': 			output_points[:,1],
		'z': 			output_points[:,2],
		'red': 		output_colors[:,0],
		'green': 	output_colors[:,1],
		'blue': 	output_colors[:,2]
		}
	cloud = PyntCloud(pd.DataFrame(data=d))
	cloud.to_file(output_file)
