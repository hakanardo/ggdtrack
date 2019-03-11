3283f218b0b693d47e71143638f5b6cc8b5198ae
----------------------------------------
-  0 Loss:  0.000261, Train Accuracy: 97.520281 %, Eval Accuracy: 97.761496 %
-  1 Loss:  0.000231, Train Accuracy: 97.882022 %, Eval Accuracy: 97.403812 %
-  2 Loss:  0.000226, Train Accuracy: 97.941973 %, Eval Accuracy: 98.264785 %
-  3 Loss:  0.000225, Train Accuracy: 97.972220 %, Eval Accuracy: 98.215840 %
-  4 Loss:  0.000225, Train Accuracy: 98.007517 %, Eval Accuracy: 98.242597 %
-  5 Loss:  0.000224, Train Accuracy: 98.015206 %, Eval Accuracy: 98.211227 %
-  6 Loss:  0.000226, Train Accuracy: 98.006311 %, Eval Accuracy: 98.326647 %
-  7 Loss:  0.000223, Train Accuracy: 98.044410 %, Eval Accuracy: 98.228054 %
-  8 Loss:  0.000225, Train Accuracy: 98.047730 %, Eval Accuracy: 98.130867 %
-  9 Loss:  0.000227, Train Accuracy: 98.043642 %, Eval Accuracy: 98.250681 %
- Result interpolated  IDF1    IDP    IDR     MOTA  MOTP num_frames
- OVERALL              82.2%  86.6%  78.1%    75.6% 0.250     112897

99c6885a50874c03028e07bd4ca1779586cabdfa
----------------------------------------
- full_run --limit 50
  - 9 Loss:  0.000354, Train Accuracy: 96.527179 %, Eval Accuracy: 95.324513 %
  
        Result
                                                   IDF1   IDP   IDR  MOTA  MOTP num_frames
        cachedir/graphs/duke_graph_3_00194195.pck 61.7% 89.9% 47.0% 46.6% 0.243        264
        cachedir/graphs/duke_graph_4_00187017.pck 83.9% 97.0% 74.0% 71.7% 0.241        600
        OVERALL                                   81.6% 96.4% 70.7% 68.7% 0.241        864
        
        Result interpolated
                                                    IDF1   IDP   IDR  MOTA  MOTP num_frames
        cachedir/graphs/duke_graph_3_00194195.pcki 68.0% 85.6% 56.4% 60.2% 0.255        264
        cachedir/graphs/duke_graph_4_00187017.pcki 96.7% 96.2% 97.2% 93.3% 0.235        600
        OVERALL                                    93.8% 95.3% 92.4% 89.4% 0.237        864
        
- fossard_run resume
  - Hamming: 32332, 31178, 26940, 30090, 28358, 27364, 26442, 25642, 27282, 27310
    
        Result
                                                   IDF1   IDP   IDR  MOTA  MOTP num_frames
        cachedir/graphs/duke_graph_3_00194195.pck 65.5% 95.0% 50.0% 47.3% 0.243        264
        cachedir/graphs/duke_graph_4_00187017.pck 80.9% 92.5% 71.9% 76.2% 0.238        600
        OVERALL                                   79.3% 92.7% 69.3% 72.7% 0.239        864
        
        Result interpolated
                                                    IDF1   IDP   IDR  MOTA  MOTP num_frames
        cachedir/graphs/duke_graph_3_00194195.pcki 95.1% 95.1% 95.1% 90.2% 0.266        264
        cachedir/graphs/duke_graph_4_00187017.pcki 85.1% 86.9% 83.3% 91.4% 0.232        600
        OVERALL                                    86.3% 87.9% 84.7% 91.2% 0.236        864
