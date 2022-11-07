"""
Script to analyze the markov chains collected in data_collection.py
"""
import os.path as op
import sys

# use local version of highway_env, before simulation you should be told this
# is a local version.
local_highway_env = op.join(op.dirname(op.realpath(__file__)), "..",)
sys.path.insert(1, local_highway_env)
import highway_env  # noqa
from highway_env.data.markov_chain import discrete_markov_chain  # noqa

# mc1 = discrete_markov_chain(num_states=2**16, transition_data=[])
# mc1.load_object("2_lane_low_density_low_cars_1000_1099")
mc2 = discrete_markov_chain(num_states=2**16, raw_data={1000: []})
mc2.load_object("2_lane_low_density_low_cars_1000_1009")
# mc3 = discrete_markov_chain(num_states=2**16, transition_data=[])
# mc3.load_object("2_lane_low_density_low_cars_1000_1001")
# mc2 = discrete_markov_chain(num_states=2**16, transition_data=[])
# mc2.load_object("2_lane_low_density_low_cars_1100_1199")
# mc3 = discrete_markov_chain(num_states=2**16, transition_data=[])
# mc3.load_object("2_lane_low_density_low_cars_1200_1299")
# mc4 = discrete_markov_chain(num_states=2**16, transition_data=[])
# mc4.load_object("2_lane_low_density_low_cars_1300_1399")
# mc5 = discrete_markov_chain(num_states=2**16, transition_data=[])
# mc5.load_object("2_lane_low_density_low_cars_1400_1499")
# mc6 = discrete_markov_chain(num_states=2**16, transition_data=[])
# mc6.load_object("2_lane_low_density_low_cars_1500_1999")
# mc7 = discrete_markov_chain(num_states=2**16, transition_data=[])
# mc7.load_object("2_lane_low_density_low_cars_2000_2999")
# mc8 = discrete_markov_chain(num_states=2**16, transition_data=[])
# mc8.load_object("2_lane_low_density_low_cars_3000_4999")

# print(mc8.transition_data[0])
print(mc2.raw_data[1000])
# print(f"1000_1099-1100_1199:{mc1.compare(mc2)}")
# print(f"1000_1199-1200_1299:{(mc1+mc2).compare(mc3)}")
# print(f"1000_1299-1300_1399:{(mc1+mc2+mc3).compare(mc4)}")
# print(f"1000_1399-1400_1499:{(mc1+mc2+mc3+mc4).compare(mc5)}")
# print(f"1000_1499-1500_1999:{(mc1+mc2+mc3+mc4+mc5).compare(mc6)}")
# print(f"1000_1999-2000_2999:{(mc1+mc2+mc3+mc4+mc5+mc6).compare(mc7)}")
# print(f"1000_2999-3000_4999:{(mc1+mc2+mc3+mc4+mc5+mc6+mc7).compare(mc8)}")
