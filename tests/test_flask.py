import unittest
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Depression Prediction</title>', response.data)
    
    def test_predict_page(self):
        response = self.client.post('/predict', data={
            'feature1': '1',  # Gender
            'feature2': '22', # Age
            'feature3': '0',  # City
            'feature4': '1',  # Profession
            'feature5': '3',  # Academic Pressure
            'feature6': '4',  # Work Pressure
            'feature7': '3.5', # CGPA
            'feature8': '3',  # Study Satisfaction
            'feature9': '2',  # Job Satisfaction
            'feature10': '6', # Sleep Duration
            'feature11': '1', # Dietary Habits
            'feature12': '1', # Degree
            'feature13': '0'  # Suicidal Thoughts
        })
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'Depression Detected' in response.data or b'No Depression' in response.data,
            "Response should contain either 'Depression Detected' or 'No Depression'"
        )

if __name__ == '__main__':
    unittest.main()
