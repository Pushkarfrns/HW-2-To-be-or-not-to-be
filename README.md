# HW-2-To-be-or-not-to-be
EECS 731:Assignment 2: HW2: To be, or not to be

Notebook: HW_2_Shakespeare_data_new.ipynb
Purpose: Deduced Additional Information and visualization

1. Loaded the Shakespeare_data.csv in a dataframe.
2. Replaced the NaN value in the Player column to be Unknown
3. Additional Information #1:
   For each Play, number of lines (PlayerLine) spoken by each Player
4. Converted the above new data (from 3) into a new data frame (playWise_lines_per_player).

5. Additional Information #2:
   To count the number of PlayerLine corresponding to each Play.
6. Converted the above new data (from 5) into a dataframe (playerLinePerPlay)
7. Plotted a graph to show: PlayerLine against Name of the Play.

8. Additional Information #3:
   Number of Players corresponding to each Play
9. Converted the above new data (from 8) into a dataframe (playersPerPlay)
10.Plotted a graph to show: Number of Players against Name of the Play.

11. Additional Information #4:
    On the basis of number of words in each PlayerLine correspondong to each Player, found the the Player that spoke the maximum number of words, and hence is the important/ or the player that has spent most time in the play. 
12. My findings --> Player named GLOUCESTER has maximum number of 14319 total words in all the PlayerLine, and hence the important/main.

13. Additional Information #5: Made a list of most frequent distinct words used in the play by their occurance. 

14. Visualization
15. In order to apply logistic regression model, changed the datatype of all attribute to int
16. Splited the data into testing and training data with the help of sklearn.model_selection import train_test_split
17. Applied logistic regression model.

Notebook: Shakespeare_notebook2.ipynb
Purpose: Logistic Regression & Visualization ( More on logistic Regression)

1. Loaded the Shakespeare_data.csv in a dataframe.
2. Handled missing values and clean the data.
3. In order to apply logistic regression model, changed the datatype of all attribute to int
4. Splited the data into testing and training data with the help of sklearn.model_selection import train_test_split
5. Applied logistic regression model.
6. Found the accuracy.
