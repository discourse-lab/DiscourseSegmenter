import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.util.List;
import java.util.ArrayList;
import org.dkpro.statistics.agreement.unitizing.UnitizingAnnotationStudy;
import org.dkpro.statistics.agreement.unitizing.KrippendorffAlphaUnitizingAgreement;

/*
 * A simple evaluation script using the Krippendorff Alpha for Unitizing as
 * implemented in the DKpro Agreement module.
 * 
 * Expects as input a tab separated data file, with one segment per line, defined
 * with four field: start index, length of segment, annotator id, category.
 * The annotator id is required to be either 0 or 1. The script is bound to a
 * one vs one evaluation.
 *
 * @author Andreas Peldszus
 */

public class UnitizingAnnotationEvaluator {

	public static void main(String[] args) throws IOException, FileNotFoundException {
		// read file and determine continuum length
		BufferedReader reader = new BufferedReader(new FileReader(args[0]));
		long continuumLength = 0;
		String line;

		List<Long> starts = new ArrayList<Long>();
		List<Long> lengths = new ArrayList<Long>();
		List<Integer> coders = new ArrayList<Integer>();
		List<String> categories = new ArrayList<String>();

		while ((line = reader.readLine()) != null) {
			if (line.startsWith("#")) {
				continue;
			}

			String[] fields = line.split("\t");
			long start = Long.parseLong(fields[0]);
			starts.add(start);
			long length = Long.parseLong(fields[1]);
			lengths.add(length);
			coders.add(Integer.parseInt(fields[2]));
			categories.add(fields[3]);

			long end = start + length;
			if (continuumLength < end) {
				continuumLength = end;
			}
		}
		reader.close();

		System.out.println("Determined continuum length of " + continuumLength);

		// initialize datastructure
		int raterCount = 2;
		UnitizingAnnotationStudy study = new UnitizingAnnotationStudy(raterCount, (int) continuumLength);
		for (int i = 0; i < starts.size(); i++) {
			study.addUnit(starts.get(i), lengths.get(i), coders.get(i), categories.get(i));
		}

		// calculate agreement
		KrippendorffAlphaUnitizingAgreement alpha = new KrippendorffAlphaUnitizingAgreement(study);
		System.out.println(alpha.calculateAgreement());
	}

}