# Model Analysis - Hopkin Network project
## BIO-210-team-9  
### Detrois Yannick, Grosjean Barbara and Houngbedji Lou

## Introduction
This summary contains the study of the Hopfield Network model we have coded. We will start by analysing the ***capacity of the model***, followed by commenting on the ***robustness***, and lastly presenting a ***visualisation*** of a network pattern retrieval. To better understand what exactly we are analysing, feel free to refer to the [README](README.md), where the general principle of a network and the project hierarchy are detailed. 


## 1. Capacity
### 1. Introduction
The first feature of our model is its ***storage capacity***, i.e. how many patterns a certain network can store. By definition, a pattern is stored if the system converges to this pattern being given a perturbed version of it as the first state of the updating sequence. An analogy to better explain this would be that if a network is able to find an image from a blurred version of this image, we consider that the network has stored the latter. We are going to study the storage capacity for both Hebbian and Storkey rules with 10 logarithmically arranged network sizes: 10, 18, 34, 63, 116, 215, 397, 733, 1354, 2500. 

To determine the capacity of the model, we test (for each network size) a range of number of starting patterns. These are sampled around the theoretical asymptotic capacity of a certain learning rule (precisely 10 values linearly chosen between 0.5 times and 2 times the theoretical capacity). After choosing a base pattern out of these starting patterns and perturbing 20% of its elements, we run the system to see if the network indeed stored this pattern. If running the system leads to convergence (thus indicating the pattern was stored) for less than 90% of the 100 trials, we consider the tested number of starting patterns out of the network's range. After running the experiment for a set of number of starting patterns, we are therefore able to assert what the maximal capacity of the network is.

Results are saved as .h5 files and are saved in the [res folder](/res) of the project. Each file contains result for one experiment in a table such as shown below for a capacity experiment using Hebbian weights and a network size of 2500. The file is named following the experiment characteristics:
- first letter: rule used for the experiment (H for Hebbian and S for Storkey)
- second letter: type of the experiment conducted (C for capacity and R for robustness)
- following digits: size of the network used for the experiment


|    |   network_size | weight_rule   |   num_patterns |   num_perturb |   match_frac |   theoretical_capacity |
|---:|---------------:|:--------------|---------------:|--------------:|-------------:|-----------------------:|
|  0 |           2500 | hebbian       |             79 |           500 |         1    |                159.764 |
|  1 |           2500 | hebbian       |            106 |           500 |         1    |                159.764 |
|  2 |           2500 | hebbian       |            133 |           500 |         0.97 |                159.764 |
|  3 |           2500 | hebbian       |            159 |           500 |         0.93 |                159.764 |
|  4 |           2500 | hebbian       |            186 |           500 |         0.67 |                159.764 |
|  5 |           2500 | hebbian       |            213 |           500 |         0.48 |                159.764 |
|  6 |           2500 | hebbian       |            239 |           500 |         0.26 |                159.764 |
|  7 |           2500 | hebbian       |            266 |           500 |         0.04 |                159.764 |
|  8 |           2500 | hebbian       |            292 |           500 |         0.02 |                159.764 |
|  9 |           2500 | hebbian       |            319 |           500 |         0    |                159.764 |


### 2. Capacity plots Hebbian weights
The plots of the percentage of retrieved patterns over the number of random base patterns (the capacity tested) for each network size and learning rule are saved [here](/plot) individually using the same naming convention than the data files. In this summary we will only present a single example of an individual plot to explain the legends and then one plot with multiple subplots for the following network sizes.

![hebbian capacity with network size 10](/plot/HC10.png)

This figure corresponds to the capacity experiment for a network of size 10 using the Hebbian rule. The x axis gives the number of patterns that the network has been initialised with and the y axis the percentage of patterns that could be retrieved for the 100 trials performed. The black curve corresponds to the experimental data while the blue vertical line is the theoretical capacity for this network size. The same colour code and axis are used in the following summary plot for all 10 network sizes.

![hebbian capacity with all network sizes subplots](/plot/SubplotHC.png)

We observe in this data two general trends. The first is that all the capacity curves are generally decreasing (this is a general trends and we do not consider fluctuations between some data points to be of any relevance). This is to be expected because as we get closer or even exceed the theoretical value, the network should less and less be able to retrieve the base pattern. The second trend is that as the network size increases, it more accurately approaches the theoretical value. The capacity for the last few network sizes in fact exactly correspond to the theoretical values, as we see a consistent retrival for a number of patterns smaller or equal to the theoretical value and then a rapid decline once exceeded. Overall, the theoretical value seems to be a good fit to the experimentatl data.


### 3. Capacity plots Storkey weights
Note that individual curves are also available for the Storkey weights. We will again show a single plot with the Storkey weights and a network size of 63 to detail the legends and then present a second figure containing subplots for all network sizes.

![storkey capacity with network size 63](/plot/SC63.png)

The axis and vertical line correspond to the same than what has been described above for the Hebbian weights. The only difference is that the experimental data curve for Storkey weights has a different colour. All other network sizes as subplots are shown in the figure below.

![storkey capacity with all network sizes subplots](/plot/SubplotSC.png)

For the Storkey weights we can observe a trend that positively correlates the normalised capacity (capacity expressed as a percentage of the theoretical capacity) and the network size. In other words, for a small network size, the real capacity is below the theoretical one, but progressively increases with the network size to then be closer to the theoretical capacity. This can be seen in the graph presented in part 4 and in the following figure in which all network sizes are superimposed, using a normalised x axis (in percent rather than network site dependant numbers) so to be able to compare retrieval results across different network sizes.

![cumulated capacities using storkey weights](/plot/CumulatedSC.png)

Using Storkey weights, the theoretical value is a less good fit for smaller network values than previously discussed for Hebbian weights. The same two trends as described for the Hebbian weights apply here aswell. However, the experimental values reach the theoretical asymptotic value for larger network sizes (in fact, only the network of size 2500 reaches it). The Storkey weights theory therefore seems to be too generous for small network sizes.


### 4. Common capacity plots and conclusion

Finally we can plot the capacity found for each size of network for Hebbian and Storkey rules. Plots for the general capacities using only one learning rule are also available in the same folder as the other plots. In the following plot is represented the capacity (number of patterns in the network for which at least 90% of the trials led to retrieval) over the size of the network for both learning rules.

![cumulated general capacity](/plot/CumulatedGeneralCapacity.png)

As the network sizes are arranged logarithmically and the theoretical capacites also are a function depending on the logarithm of the network's size, it came in handy to use logarithmic scales for both axes.

![cumulated general capacity log scale](/plot/CumulatedGeneralCapacityLog.png)

These two figures confirm the trends described earlier in the sections specific to a single learning rule. Greater network sizes correlate with increasingly close theoretical and experimental data. For Storkey weights however, it takes until a larger network size until the theoretical capacity is reached, for smaller network sizes the empiric values are below than what theory gives. For Hebbian weights, empiric values are always very close to the expected values, almost completly overlapping with the theoretical values. Although for both curves experimental data seems to follow the theoretical curve, the fit is better using Hebbian weights for small network sizes. Finally, we can see that as expected and predicted by theory, the general capacity is greater for using the Storkey rule compared to the Hebbian rule.


## 2. Robustness
### 1. Introduction
In this part we aim to determine how much we can perturb a base pattern until retrieval of the base pattern is no longer reached. This defines the ***robustness*** of the model. To observe this second feature of our model we start by perturbing 20% of a pattern's elements and then proceed to increase the percentage of perturbation by steps of 5% until updating does not anymore lead to retrieval of the original pattern in at least 90% of the 100 trials. We again use 10 different network sizes : 10, 18, 34, 63, 116, 215, 397, 733, 1354, 2500, and choose a network capacity for which we know the system converges for a perturbation of 20%. This capacity corresponds in our case to 0.75 times the theoretical asymptotic value.

Experimental results are saved here aswell as .h5 files with the same naming convention in the [res folder](/res), and contain information as tables such as this example for a robustness experiment (R) using Storkey weights (S) and a network size of 2500.

|    |   network_size | weight_rule   |   num_patterns |   num_perturb |   match_frac |
|---:|---------------:|:--------------|---------------:|--------------:|-------------:|
|  0 |           2500 | storkey       |            473 |           500 |         1    |
|  1 |           2500 | storkey       |            473 |           625 |         1    |
|  2 |           2500 | storkey       |            473 |           750 |         0.94 |
|  3 |           2500 | storkey       |            473 |           875 |         0    |
|  4 |           2500 | storkey       |            473 |           999 |         0    |
|  5 |           2500 | storkey       |            473 |          1125 |         0    |
|  6 |           2500 | storkey       |            473 |          1249 |         0    |
|  7 |           2500 | storkey       |            473 |          1374 |         0    |
|  8 |           2500 | storkey       |            473 |          1499 |         0    |
|  9 |           2500 | storkey       |            473 |          1624 |         0    |
| 10 |           2500 | storkey       |            473 |          1750 |         0    |
| 11 |           2500 | storkey       |            473 |          1874 |         0    |


### 2. Robustness curves Hebbian weights

The plots shown are the fraction of retrieved patterns over the percentage of perturbations for each network size using Hebbian weights. All separate plots are available [here](/plot) but we will here only show and discuss a general figure with all the subplots:

![hebbian robusntess with network size 2500](/plot/SubplotHR.png)

For robustness aswell two general trends can be seen. The first one is very similar to the corresponding capacity trend, which is that retreival decreases when the studied feature (here the robustness) is increased, i.e. a greater percentage of the base pattern is perturbed. It is normal and expected for a network not to recognize a pattern anymore once it has changed too much. Secondly, we notice that the robustness seems to consistently stabilise at 40% perturbation for larger network sizes. The boundary between retrived and not retreived is already clearer for larger network sizes, as retrieval drops much more drastically once the maximal robustness passed (the smaller sized networks' retrieval percentages decrease more progressively). 

### 3. Robustness curves Storkey weights

Similarly to the robusntess curves using Hebbian weights we obtain 10 subplots for the networks using the Storkey weights.

![hebbian robustness with network size 2500](/plot/SubplotSR.png)

Superimposing these 10 plots for the Storkey weights again shows an increasing tendency: larger networks seem to be more robust, for the 10 tested network sizes at least.

![cumulated robustness using storkey weights](/plot/CumulatedSR.png)

We can in this figure observe clusters with the larger network sizes having robustness plots that are shifted to the right (and have much steeper curve drop), which means they can generally handle more perturbation. As seen earlier for the Storkey rule capacity analysis, the network gets closer to theoretical values when its size gets larger. This would also explain why robustness increases with an increasing network size: for small network sizes, robustness is under the asymptotic theoretical curves, and once reached this curve by increasing the network size, both capacity and robusntess stabilise. To confirm this we would however need more experimental data for larger network sizes. Once again the same overall trends can be observed for Hebbian and Storkey weights, but the performance using Storkey weights is again shifter towards bigger networks. That is, a certain robustness (or generally any curve trend) is reached the same way as for the Hebbian weights but for greater network sizes.


### 4. Common robustness curves and conclusion

For robustness too we are now able to compare both learning rule's performance. To do so we superimposed the robustness (maximal percentual perturbation to attain 80% retrieval along all the 100 trials for each experiment) for both learning rule in the following figure.

![cumulated general robustness](/plot/CumulatedGeneralRobustness.png)

To have an equal space between the network sizes and therefore better see the performance for smaller network sizes, we used a logarithmic scale.

![cumulated general robustness log scale](/plot/CumulatedGeneralRobustnessLog.png)

These plots confirm the overall trends presented above, except for a few outliers. There is, even though only slight, an increase of robustness for a growing network size. This increase is not constant and seems to flatten out for larger network sizes. Lastly, we can remark that the Hebbian weights are overall much more resistant to pattern perturbation for the studied network sizes (with an average of around 25% for all network sizes using Storkey weights against around 35% for the same network sizes using Storkey weights). The robustness difference between both learning rules indeed is of at least around 5% (ignoring one outlier for the Hebbian weights). 

In conclusion, Storkey weights seem to be less resiliant to pattern perturbation for the studied network sizes and are also outranked by theoretical capacity values. These values are however greater than the corresponding ones for Hebbian weights, meaning that although less consistent, the capacity of a Storkey network is greater. Although having a smaller capacity, the Hebbian model showed more consistency, being closer to theoretical values and more robust to a higher perturbation.  


## 3. Examples
### 1. Image retrieval from a corner of the image
The first example visualises the network's memory property by displaying the perturbed pattern followed by the network has been able to find (and some intermediate states to show the evolution of the system). The base pattern can be generated from any picture, in this particular case, the chosen image is a view on the most notorious mountain of Switzerland, the Matterhorn ([Matterhorn initial picture](/images/mountain1.jpg)).

This image is downsampled to a size that is manageable for our network and then turned into a black and white picture, before perturbation and updating. Perturbation consists in turning all the image white except for a corner that is left untouched. The corner left represents 36% of the total area, which does **not** mean that 64% of the image have been perturbed, because not all the pixels of the blanked out area were black. A few states along the updating are displayed in the following figure. 

![Matterhorn example](/animations/Matterhorn.png)

A GIF animation shows the network in action:

![Matterhorn example](/animations/Matterhorn.gif)


### 2. Image retrieval from a picture similar to the memorised one
This second example shows how a network is able to find a computer-made writing (in other words the "perfect" version of the writing) starting with the same text, but hand-written (which is more irregular and flawed). We considered this to be another interesting and perhaps useful feature of a neuronal network simulation and therefore added it to this summary.

![123.png](/animations/123.png)

This GIF animation shows the network's updating sequence interactively: 

![123 example](/animations/123.gif)
