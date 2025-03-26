#  Copyright (c) 2025. Pedro Encarnação . Universidade de Aveiro LICENSE: CC BY-NC-SA 4.0 # ****************************
#

# *******************************************************
# * FILE: radioisotopeTest
# * AUTHOR: Pedro Encarnação
# * DATE: 26/03/2025
# * LICENSE: "CC BY-NC-SA 4.0"
# *******************************************************

import unittest
from src.TORFilesReader import RadioisotopeInfo


class TestRadioisotopeInfo(unittest.TestCase):
    def setUp(self):
        """Set up a new RadioisotopeInfo instance before each test."""
        self.radioisotope = RadioisotopeInfo()

    def test_set_and_get_tracers(self):
        """Test setting and retrieving tracers."""
        tracers = ['F18', 'C11']
        self.radioisotope.setTracers(tracers)
        self.assertEqual(self.radioisotope.tracers, tracers)

    def test_set_and_get_half_lifes(self):
        """Test setting and retrieving half-lives with valid input."""
        self.radioisotope.setTracers(['F18', 'C11'])
        half_lives = [6586.26, 1224.0]
        self.radioisotope.setHalfLifes(half_lives)
        self.assertEqual(self.radioisotope.halfLifes, half_lives)

    def test_set_half_lifes_invalid_length(self):
        """Test setting half-lives with incorrect length compared to tracers."""
        self.radioisotope.setTracers(['F18', 'C11'])
        with self.assertRaises(ValueError):
            self.radioisotope.setHalfLifes([6586.26])  # Should raise an error

    def test_set_half_lifes_invalid_value(self):
        """Test setting half-lives with negative values."""
        self.radioisotope.setTracers(['F18'])
        with self.assertRaises(ValueError):
            self.radioisotope.setHalfLifes([-5000.0])  # Negative value should raise an error

    def test_set_half_lifes_invalid_type(self):
        """Test setting half-lives with non-float values."""
        self.radioisotope.setTracers(['F18'])
        with self.assertRaises(ValueError):
            self.radioisotope.setHalfLifes(["6586"])  # String instead of float

    def test_set_and_get_positron_fractions(self):
        """Test setting and retrieving positron fractions with valid input."""
        positron_fractions = [0.97, 0.89]
        self.radioisotope.setPositronFractions(positron_fractions)
        self.assertEqual(self.radioisotope.positronFractions, positron_fractions)

    def test_set_positron_fractions_invalid_value(self):
        """Test setting positron fractions outside the 0-1 range."""
        with self.assertRaises(ValueError):
            self.radioisotope.setPositronFractions([1.5])  # Out of range

    def test_set_positron_fractions_invalid_type(self):
        """Test setting positron fractions with non-float values."""
        with self.assertRaises(ValueError):
            self.radioisotope.setPositronFractions(["0.97"])  # String instead of float

    def test_set_and_get_decay_types(self):
        """Test setting and retrieving valid decay types."""
        decay_types = ["BetaPlus", "Alpha"]
        self.radioisotope.setDecayTypes(decay_types)
        self.assertEqual(self.radioisotope.decayTypes, decay_types)

    def test_set_decay_types_invalid_value(self):
        """Test setting decay types with an invalid value."""
        with self.assertRaises(ValueError):
            self.radioisotope.setDecayTypes(["UnknownDecay"])  # Not a valid decay type

    def test_set_and_get_decay_energies(self):
        """Test setting and retrieving decay energies with valid input."""
        decay_energies = [511.0, 1200.5]
        self.radioisotope.setDecayEnergies(decay_energies)
        self.assertEqual(self.radioisotope.decayEnergies, decay_energies)

    def test_set_decay_energies_invalid_type(self):
        """Test setting decay energies with a non-float value."""
        with self.assertRaises(ValueError):
            self.radioisotope.setDecayEnergies([511, "1200"])  # String in list

    def test_set_and_get_route(self):
        """Test setting and retrieving route information."""
        route = ["IV"]
        self.radioisotope.setRoute(route)
        self.assertEqual(self.radioisotope.route, route)

    def test_set_route_invalid_type(self):
        """Test setting route with an invalid type."""
        with self.assertRaises(ValueError):
            self.radioisotope.setRoute("IV")  # Should be a list

if __name__ == '__main__':
    unittest.main()