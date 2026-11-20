import importlib.util
import unittest

import numpy as np

from opendleto import compare, plot_tensor


@unittest.skipUnless(importlib.util.find_spec("plotly"), "Plotly lab extra is not installed")
class VisualizationTests(unittest.TestCase):
    def test_plot_tensor_filters_complex_magnitudes(self):
        tensor = np.zeros((2, 2, 2), dtype=complex)
        tensor[0, 1, 0] = 3 + 4j
        tensor[1, 0, 1] = 0.1j
        figure = plot_tensor(tensor, tol=0.2, title="Example", color="red")
        trace = figure.data[0]
        self.assertEqual(trace.type, "scatter3d")
        self.assertEqual(list(trace.x), [0])
        self.assertEqual(list(trace.y), [1])
        self.assertEqual(list(trace.z), [0])
        self.assertEqual(list(trace.customdata), [5])
        self.assertEqual(trace.marker.color, "red")
        self.assertEqual(figure.layout.title.text, "Example")
        self.assertEqual(tuple(figure.layout.scene.xaxis.range), (-0.5, 1.5))
        self.assertEqual(tuple(figure.layout.scene.yaxis.range), (-0.5, 1.5))
        self.assertEqual(tuple(figure.layout.scene.zaxis.range), (-0.5, 1.5))

    def test_compare_uses_separate_scenes_and_handles_empty_tensor(self):
        figure = compare(np.ones((7, 6, 5)), np.zeros((7, 6, 5)),
                         left_title="Control", right_title="Experiment")
        self.assertEqual([trace.scene for trace in figure.data], ["scene", "scene2"])
        self.assertEqual(len(figure.data[1].x), 0)
        self.assertEqual(tuple(figure.layout.scene2.xaxis.range), (-0.5, 6.5))
        self.assertEqual(tuple(figure.layout.scene2.yaxis.range), (-0.5, 5.5))
        self.assertEqual(tuple(figure.layout.scene2.zaxis.range), (-0.5, 4.5))
        self.assertEqual([a.text for a in figure.layout.annotations], ["Control", "Experiment"])
        with self.assertRaisesRegex(ValueError, "three axes"):
            plot_tensor(np.ones((2, 2)))

    def test_plot_tensor_does_not_trim_nonzero_entries(self):
        tensor = np.ones((11, 11, 11))
        figure = plot_tensor(tensor)
        self.assertEqual(len(figure.data[0].x), tensor.size)
        self.assertEqual(tuple(figure.layout.scene.xaxis.range), (-0.5, 10.5))
        self.assertFalse(figure.layout.scene.xaxis.autorange)


if __name__ == "__main__":
    unittest.main()
