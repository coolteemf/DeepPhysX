if __name__ == '__main__':

    import os
    from DeepPhysX_Core.Visualizer.SampleVisualizer import SampleVisualizer

    folder = os.path.dirname(os.path.abspath(__file__))
    visualizer = SampleVisualizer(folder)
