# prediction_logic.py
"""
IQsmartBot ke liye 37-layer Prediction Logic System ka base structure.
Ye file PRNG number analysis, trend detection, feedback integration, 
market support/resistance, aur smart money management handle karegi.

Har layer ka simple function yahan define hai.
"""

from collections import Counter
import numpy as np

class PredictionLogic:
    def __init__(self):
        # Yahan apne initial weights, thresholds, aur variables rakh sakte ho
        self.feedback_weights = {}  # Feedback based weights update ke liye
        self.session_data = []      # Current session ke PRNG numbers store karne ke liye
        self.big_threshold = 5      # 0-4 Small, 5-9 Big
        self.red_numbers = {0,2,4,6,8}
        self.green_numbers = {1,3,5,7,9}

    # 1. Multi-Dimensional Analysis (example: simple statistical summary)
    def multi_dimensional_analysis(self, numbers):
        mean = np.mean(numbers)
        median = np.median(numbers)
        mode = Counter(numbers).most_common(1)[0][0]
        return {'mean': mean, 'median': median, 'mode': mode}

    # 2. Probability Distribution - Number frequency percentage
    def probability_distribution(self, numbers):
        count = Counter(numbers)
        total = len(numbers)
        prob_dist = {num: count[num]/total for num in count}
        return prob_dist

    # 3. Trend Identification (Big/Small + Red/Green count difference)
    def trend_identification(self, numbers):
        big_count = sum(1 for n in numbers if n >= self.big_threshold)
        small_count = len(numbers) - big_count
        red_count = sum(1 for n in numbers if n in self.red_numbers)
        green_count = len(numbers) - red_count

        big_small_trend = 'Big' if big_count > small_count else 'Small'
        red_green_trend = 'Red' if red_count > green_count else 'Green'

        return {'big_small_trend': big_small_trend, 'red_green_trend': red_green_trend}

    # 4. Recent Data Weighting (weight recent numbers zyada)
    def recent_data_weighting(self, numbers, window=10):
        recent_numbers = numbers[-window:]
        weighted_sum = sum(recent_numbers) * 1.5  # Example weight
        return weighted_sum / window

    # 5. Big-Small Classification for each number
    def big_small_classify(self, number):
        return 'Big' if number >= self.big_threshold else 'Small'

    # 6. Red/Green Classification for each number
    def red_green_classify(self, number):
        if number in self.red_numbers:
            return 'Red'
        elif number in self.green_numbers:
            return 'Green'
        else:
            return 'Unknown'

    # 7. Dual Verification (check if big_small and red_green trend match some pattern)
    def dual_verification(self, numbers):
        trend = self.trend_identification(numbers)
        # Simple check: Big trend must correspond to Green trend for confirmation (example logic)
        if trend['big_small_trend'] == 'Big' and trend['red_green_trend'] == 'Green':
            return True
        else:
            return False

    # 8. Risk Adjustment (dummy example based on variance)
    def risk_adjustment(self, numbers):
        variance = np.var(numbers)
        if variance > 8:
            return 'High Risk'
        elif variance > 4:
            return 'Medium Risk'
        else:
            return 'Low Risk'

    # 9. Dynamic Thresholds (example adjust big_threshold based on mean)
    def dynamic_thresholds(self, numbers):
        mean = np.mean(numbers)
        if mean > 5:
            self.big_threshold = 6
        else:
            self.big_threshold = 5
        return self.big_threshold

    # 10. Confidence Score Calculation (simple example)
    def confidence_score(self, numbers):
        prob = self.probability_distribution(numbers)
        score = sum(prob.values()) * 100 / len(prob)  # dummy % score
        return min(100, max(0, score))

    # 11. Clustering Patterns Analysis (basic clusters by number proximity)
    def clustering_patterns(self, numbers):
        clusters = {}
        for num in numbers:
            key = num // 3  # groups 0-2,3-5,6-8,9
            clusters.setdefault(key, []).append(num)
        return clusters

    # 12. Streak Detection (find longest streak of big or small)
    def streak_detection(self, numbers):
        max_streak = 1
        current_streak = 1
        last_class = self.big_small_classify(numbers[0])
        for num in numbers[1:]:
            current_class = self.big_small_classify(num)
            if current_class == last_class:
                current_streak += 1
                if current_streak > max_streak:
                    max_streak = current_streak
            else:
                current_streak = 1
            last_class = current_class
        return max_streak

    # 13. Feedback-Based Weighting (update weights based on feedback)
    def update_weights_feedback(self, feedback_number, positive=True):
        if positive:
            self.feedback_weights[feedback_number] = self.feedback_weights.get(feedback_number, 0) + 1
        else:
            self.feedback_weights[feedback_number] = self.feedback_weights.get(feedback_number, 0) - 1

    # 14. Anomaly Detection (simple outlier detection using z-score)
    def anomaly_detection(self, numbers):
        mean = np.mean(numbers)
        std = np.std(numbers)
        anomalies = [n for n in numbers if abs(n - mean) > 2*std]
        return anomalies

    # 15. Adaptive Learning Rate (dummy example)
    def adaptive_learning_rate(self, iteration):
        base_rate = 0.1
        rate = base_rate / (1 + 0.01 * iteration)
        return rate

    # 16. Live Data Weighting (weight recent input heavier)
    def live_data_weighting(self, numbers):
        weights = np.linspace(0.1, 1, len(numbers))
        weighted_avg = np.average(numbers, weights=weights)
        return weighted_avg

    # 17. Historical Cross-Check (check previous sessions data)
    def historical_cross_check(self, previous_numbers, current_numbers):
        overlap = set(previous_numbers) & set(current_numbers)
        return len(overlap)

    # 18. Bayesian Updating (simple Bayesian update example)
    def bayesian_update(self, prior, likelihood):
        posterior = (prior * likelihood) / ((prior * likelihood) + ((1 - prior) * (1 - likelihood)))
        return posterior

    # 19. Smart Money Management (suggest capital increase or stop)
    def smart_money_management(self, capital, win_rate):
        if win_rate > 0.7:
            return min(capital * 2, 300)  # Max 300
        elif win_rate < 0.3:
            return max(capital / 2, 100)  # Min 100
        else:
            return capital

    # 20. Risk-Reward Analysis (dummy ratio)
    def risk_reward_analysis(self, risk, reward):
        if reward == 0:
            return 0
        return reward / risk

    # 21. Stop Loss Zone Protection (dummy zone example)
    def stop_loss_zone(self, current_loss, stop_loss_limit=50):
        if current_loss >= stop_loss_limit:
            return True  # Stop loss hit
        return False

    # 22. Red/Green Block Repetition Check (simple repeat check)
    def red_green_block_repeat(self, colors):
        # colors list example: ['Red', 'Green', 'Red', 'Red']
        repeats = sum(1 for i in range(1, len(colors)) if colors[i] == colors[i-1])
        return repeats

    # 23. Last 30 Rounds Consistency Analysis (example on last 30)
    def last_30_rounds_consistency(self, numbers):
        last_30 = numbers[-30:] if len(numbers) >= 30 else numbers
        return self.trend_identification(last_30)

    # 24. High Probability Cycle Return Detector (dummy cycle detect)
    def high_prob_cycle_detector(self, numbers):
        # Simple pattern detection example
        patterns = [numbers[i:i+3] for i in range(len(numbers)-2)]
        repeats = sum(1 for p in patterns if p == patterns[0])
        return repeats

    # 25. Big-Small Pattern Flip Alert (detect flip)
    def pattern_flip_alert(self, numbers):
        flips = 0
        for i in range(1, len(numbers)):
            if self.big_small_classify(numbers[i]) != self.big_small_classify(numbers[i-1]):
                flips += 1
        return flips

    # 26. Dominance Pattern Detector (color/size)
    def dominance_pattern_detector(self, numbers):
        big_count = sum(1 for n in numbers if n >= self.big_threshold)
        red_count = sum(1 for n in numbers if n in self.red_numbers)
        green_count = len(numbers) - red_count
        if big_count > len(numbers)*0.6:
            return "Big Dominance"
        if red_count > green_count:
            return "Red Dominance"
        else:
            return "Green Dominance"

    # 27. Repeating Group Alert System (dummy example)
    def repeating_group_alert(self, numbers):
        groups = [numbers[i:i+2] for i in range(len(numbers)-1)]
        repeats = sum(1 for i in range(1, len(groups)) if groups[i] == groups[i-1])
        return repeats

    # 28. Multi-Point Overlap Signal Filter (dummy overlap)
    def multi_point_overlap_filter(self, numbers):
        unique = set(numbers)
        return len(numbers) - len(unique)

    # 29. Error Trend Analysis Module (dummy error rate)
    def error_trend_analysis(self, errors, total):
        if total == 0:
            return 0
        return errors / total

    # 30. Confidence Deviation Engine (dummy deviation)
    def confidence_deviation(self, scores):
        return np.std(scores)

    # 31. Recursive Correction Engine (dummy correction)
    def recursive_correction(self, prediction, feedback):
        corrected = prediction + 0.1 * (feedback - prediction)
        return corrected

    # 32. Model Comparison & A/B Testing Module (dummy example)
    def model_comparison(self, model_a_score, model_b_score):
        return "Model A" if model_a_score > model_b_score else "Model B"

    # 33. Feedback Integration Module (adjust based on feedback)
    def feedback_integration(self, current_score, feedback):
        if feedback:
            return current_score + 1
        else:
            return current_score - 1

    # 34. Real-Time Adaptive Weighting Engine (dummy weight adjust)
    def adaptive_weighting(self, weight, feedback):
        if feedback:
            return weight * 1.1
        else:
            return weight * 0.9

    # 35. Live Anomaly and Trend Detector (dummy)
    def live_anomaly_trend(self, numbers):
        anomalies = self.anomaly_detection(numbers)
        trend = self.trend_identification(numbers)
        return {'anomalies': anomalies, 'trend': trend}

    # 36. User-Driven Parameter Tuner (placeholder)
    def user_parameter_tuner(self, param, value):
        # Example: Adjust big_threshold by user input
        if param == 'big_threshold':
            self.big_threshold = value
        return self.big_threshold

    # 37. Meta-Analytics & Global Pattern Aggregator (dummy aggregator)
    def meta_analytics(self, sessions_data):
        all_numbers = []
        for session in sessions_data:
            all_numbers.extend(session)
        common = Counter(all_numbers).most_common(5)
        return common


# Baby-level usage example:
if __name__ == "__main__":
    pl = PredictionLogic()
    sample_numbers = [6,3,2,8,2,9,4,2,8,3,5,7,9,1]
    print("Multi-Dimensional Analysis:", pl.multi_dimensional_analysis(sample_numbers))
    print("Trend Identification:", pl.trend_identification(sample_numbers))
    print("Confidence Score:", pl.confidence_score(sample_numbers))
