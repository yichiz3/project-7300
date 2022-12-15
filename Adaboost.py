import numpy as np
import random
from sklearn.metrics import accuracy_score

class Adaboost:
    def __init__(self, num_learner: int, Model, random_State=None, 
                 min_samples_split_lower=2, min_samples_split_upper=5, max_depth_lower=2, max_depth_upper=6):
        self.num_learner = num_learner
        self.entry_weights = None
        self.learner_weights = None
        self.sorted_learners = None
        self.map_Label_2_Index = {}
        self.learners = self.create_Learners(
            Model=Model,
            random_State=random_State,
            min_samples_split_lower=min_samples_split_lower,
            min_samples_split_upper=min_samples_split_upper,
            max_depth_lower=max_depth_lower,
            max_depth_upper=max_depth_upper)

    def create_Learners(self, Model, random_State, min_samples_split_lower=2, min_samples_split_upper=5, 
                        max_depth_lower=2, max_depth_upper=6):
        if random_State:
            random.setstate(random_State)
        learners = []
        for i in range(self.num_learner):
            learners.append(
                Model(
                    min_samples_split=random.randint(
                        min_samples_split_lower,
                        min_samples_split_upper),
                    max_depth=random.randint(
                        max_depth_lower,
                        max_depth_upper
                    )))
        return learners

    def __fit_Learners(self, X_Train, Y_Train):
        for learner in self.learners:
            learner.fit(X_Train, Y_Train)

    def fit(self, X_Train, Y_Train):
        self.num_Of_Classes = len(np.unique(Y_Train))
        self.map_Label_2_Index = {
            label: index for index, label in enumerate(np.unique(Y_Train))}
        self.map_Index_2_Label = {
            index: label for index, label in enumerate(np.unique(Y_Train))}

        num_Of_Data = X_Train.shape[0]
        self.__fit_Learners(X_Train=X_Train, Y_Train=Y_Train)
        self.entry_weights = np.full(
            (num_Of_Data,), fill_value=1/num_Of_Data, dtype=np.float32)
        self.learner_weights = np.zeros((self.num_learner,), dtype=np.float32)

        score = [0 for i in range(self.num_learner)]
        for learner_idx, learner in enumerate(self.learners):
            score[learner_idx] = accuracy_score(
                learner.predict(X_Train), Y_Train)
        self.sorted_learners = [l for l, e in sorted(
            zip(self.learners, score), key=lambda pair: pair[1], reverse=True)]

        for learner_idx, learner in enumerate(self.sorted_learners):
            Y_Predicted = learner.predict(X_Train)
            is_wrong = np.array(Y_Predicted != Y_Train.reshape(-1)).astype(int)
            weighted_learner_error = np.sum(
                is_wrong * self.entry_weights)/self.entry_weights.sum()
            self.learner_weights[learner_idx] = max(0, 
                    np.log(1/(weighted_learner_error + 1e-6) - 1) + np.log(
                self.num_Of_Classes - 1))
            alpha_arr = np.full(
                (num_Of_Data,), fill_value=self.learner_weights[learner_idx], 
                                                dtype=np.float32)
            self.entry_weights = self.entry_weights * \
                np.exp(alpha_arr * is_wrong)
            self.entry_weights = self.entry_weights/self.entry_weights.sum()

        self.learner_weights = self.learner_weights/self.learner_weights.sum()

    def predict(self, features):
        return [self.predict_One(feature) for feature in features]

    def predict_One(self, feature):
        pooled_prediction = np.zeros((self.num_Of_Classes,), dtype=np.float32)
        for learner_idx, learner in enumerate(self.sorted_learners):
            predicted_cat = learner.predict_One(feature)
            prediction = np.full(
                (self.num_Of_Classes,), fill_value=-1/(self.num_Of_Classes-1), 
                                                        dtype=np.float32)
            prediction[self.map_Label_2_Index[predicted_cat]] = 1
            pooled_prediction += prediction*self.learner_weights[learner_idx]

        return self.map_Index_2_Label[np.argmax(pooled_prediction)]
