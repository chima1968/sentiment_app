import unittest
import pandas as pd
import numpy as np
from classifier_analysis import augment_text  # Import the function you want to test

class TestTextAugmentation(unittest.TestCase):

    def setUp(self):
        # This method will be called before each test
        self.sample_texts = [
            "This is a sample text for augmentation.",
            "Another example of text that can be used."
        ]

    def test_augment_text_output_type(self):
        text = "Test text for augmentation."
        result = augment_text(text, self.sample_texts)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # Assuming num_augmentations=2

    def test_augment_text_not_empty(self):
        text = "Short text."
        result = augment_text(text, self.sample_texts)
        for augmented_text in result:
            self.assertTrue(len(augmented_text) > 0)

    def test_augment_text_different_from_original(self):
        text = "This is the original text."
        result = augment_text(text, self.sample_texts)
        for augmented_text in result:
            self.assertNotEqual(text, augmented_text)

    def test_augment_text_with_empty_input(self):
        text = ""
        result = augment_text(text, self.sample_texts)
        self.assertEqual(len(result), 2)  # Should still return 2 augmentations
        for augmented_text in result:
            self.assertEqual(augmented_text, "")

    def test_augment_text_with_empty_samples(self):
        text = "Test text."
        empty_samples = []
        result = augment_text(text, empty_samples)
        self.assertEqual(len(result), 2)
        for augmented_text in result:
            self.assertEqual(augmented_text, text)  # Should return original text if no samples

    def test_dataframe_augmentation(self):
        df = pd.DataFrame({
            'text': ['Text 1', 'Text 2', 'Text 3', 'Text 4', 'Text 5']
        })
        sample_texts = df['text'].sample(n=2).tolist()
        df['augmented'] = df['text'].apply(lambda x: augment_text(x, sample_texts) if x not in sample_texts else [x])
        df = df.explode('augmented')
        
        self.assertEqual(len(df), 11)  # 5 original + 6 augmented (3 texts * 2 augmentations)
        self.assertTrue('augmented' in df.columns)

if __name__ == '__main__':
    unittest.main()