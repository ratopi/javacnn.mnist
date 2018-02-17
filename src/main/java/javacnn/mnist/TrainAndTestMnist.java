package javacnn.mnist;

import java.io.File;
import java.io.IOException;
import java.util.Iterator;

import javacnn.cnn.CNN;
import javacnn.cnn.Layer;
import javacnn.dataset.Dataset;
import javacnn.util.Util;

public class TrainAndTestMnist
{
	public static void main( final String[] args ) throws IOException
	{
		final CNN.LayerBuilder builder = new CNN.LayerBuilder();

		builder.addLayer( Layer.buildInputLayer( new Layer.Size( 28, 28 ) ) );
		builder.addLayer( Layer.buildConvLayer( 6, new Layer.Size( 5, 5 ) ) );
		builder.addLayer( Layer.buildSampLayer( new Layer.Size( 2, 2 ) ) );
		builder.addLayer( Layer.buildConvLayer( 12, new Layer.Size( 5, 5 ) ) );
		builder.addLayer( Layer.buildSampLayer( new Layer.Size( 2, 2 ) ) );
		builder.addLayer( Layer.buildOutputLayer( 10 ) );

		final CNN cnn = new CNN( builder, 50 );

		final Dataset trainset = readDataset( "train" );
		final Dataset testset = readDataset( "t10k" );

		cnn.train( trainset, 10 );

		System.out.println();
		System.out.println( "now testing ..." );

		int total = 0;
		int correct = 0;

		// CNN cnn = CNNLoader.loadModel(modelName);
		final Iterator<Dataset.Record> iterator = testset.iterator();
		while ( iterator.hasNext() )
		{
			Layer.prepareForNewBatch();
			final Dataset.Record record = iterator.next();
			final double[] propagate = cnn.propagate( record );
			final int label = Util.getMaxIndex( propagate );

			if ( label == record.getLabel() ) correct++;

			total++;
		}

		System.out.println( correct + "/" + total + "=" + ( (double) correct / total ) );
	}

	private static Dataset readDataset( final String prefix ) throws IOException
	{
		final File imagesFile = new File( "mnist/" + prefix + "-images-idx3-ubyte.gz" );
		final File labelsFile = new File( "mnist/" + prefix + "-labels-idx1-ubyte.gz" );

		final ReadMnist readMnist = new ReadMnist( labelsFile, imagesFile );

		return readMnist.createDataset();
	}
}
