package javacnn.mnist;

import java.io.File;
import java.io.IOException;

import de.ratopi.mnist.read.MnistReader;

import javacnn.dataset.Dataset;

public class ReadMnist
{
	private final File labelsFile;
	private final File imagesFile;

	public ReadMnist( final File labelsFile, final File imagesFile )
	{
		this.imagesFile = imagesFile;
		this.labelsFile = labelsFile;
	}

	public Dataset createDataset() throws IOException
	{
		final MnistReader mnistReader = new MnistReader( labelsFile, imagesFile );

		final Dataset dataset = new Dataset();

		final MnistReader.DataArrayImageHandler dataArrayImageHandler = new DatasetDataArrayImageHandler( dataset );

		mnistReader.handleAllRemaining( dataArrayImageHandler );

		return dataset;
	}

	// ---

	private class DatasetDataArrayImageHandler implements MnistReader.DataArrayImageHandler
	{
		private Dataset dataset;

		public DatasetDataArrayImageHandler( final Dataset dataset )
		{
			this.dataset = dataset;
		}

		@Override
		public void handle( final long index, final byte[] dataIn, final byte labelIn )
		{
			final double[] data = new double[ dataIn.length ];
			final double label = ( (int) labelIn ) & 0xFF;

			for ( int i = 0; i < dataIn.length; i++ )
			{
				data[ i ] = ( (int) dataIn[ i ] ) & 0xFF;
			}

			dataset.append( data, label );
		}
	}
}
