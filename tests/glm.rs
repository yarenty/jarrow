use datafusion_log_reader::LogReader;

#[cfg(test)]
mod log_reader_test {

    use crate::LogReader;
    use arrow::array::RecordBatchReader;

    #[test]
    fn should_not_exist() {
        let package_dir = env!("CARGO_MANIFEST_DIR");
        let path = format!("{package_dir}/data_not_exits.log");

        let reader = LogReader::new(&path);

        assert!(reader.is_err());
    }

    #[test]
    fn should_read_all() {
        let package_dir = env!("CARGO_MANIFEST_DIR");
        let path = format!("{package_dir}/tests/data/test.log");

        let mut reader = LogReader::new(&path).unwrap();

        let _schema = reader.schema();
        let result = reader.next().unwrap().unwrap();

        assert_eq!(4, result.num_columns());
        assert_eq!(15, result.num_rows());
    }
}

#[cfg(test)]
#[ctor::ctor]
fn init() {
    // Enable RUST_LOG logging configuration for test
    let _ = env_logger::builder().is_test(true).try_init();
}

/*
  public static void testScoring(GLMModel m, Frame fr) {
    Scope.enter();
    // try scoring without response
    Frame fr2 = new Frame(fr);
    fr2.remove(m._output.responseName());
//    Frame preds0 = Scope.track(m.score(fr2));
//    fr2.add(m._output.responseName(),fr.vec(m._output.responseName()));
    // standard predictions
    Frame preds = Scope.track(m.score(fr2));
    m.adaptTestForTrain(fr2,true,false);
    fr2.remove(fr2.numCols()-1); // remove response
    int p = m._output._dinfo._cats + m._output._dinfo._nums;
    int p2 = fr2.numCols() - (m._output._dinfo._weights?1:0)- (m._output._dinfo._offset?1:0);
    assert p == p2: p + " != " + p2;
    fr2.add(preds.names(),preds.vecs());
    // test score0
    new TestScore0(m,m._output._dinfo._weights,m._output._dinfo._offset).doAll(fr2);
    // test pojo
    if((!m._output._dinfo._weights && !m._output._dinfo._offset))
      Assert.assertTrue(m.testJavaScoring(fr,preds,1e-15));
    Scope.exit();
  }


  // class to test score0 since score0 is now not being called by the standard bulk scoring
  public static class TestScore0 extends MRTask {
    final GLMModel _m;
    final boolean _weights;
    final boolean _offset;

    public TestScore0(GLMModel m, boolean w, boolean o) {_m = m; _weights = w; _offset = o;}

    private void checkScore(long rid, double [] predictions, double [] outputs){
      int start = 0;
      if(_m._parms._family == Family.binomial && Math.abs(predictions[2] - _m.defaultThreshold()) < 1e-10)
        start = 1;
      if(_m._parms._family == Family.multinomial) {
        double [] maxs = new double[2];
        for(int j = 1; j < predictions.length; ++j) {
          if(predictions[j] > maxs[0]) {
            if(predictions[j] > maxs[1]) {
              maxs[0] = maxs[1];
              maxs[1] = predictions[j];
            } else maxs[0] = predictions[j];
          }
        }
        if((maxs[1] - maxs[0]) < 1e-10)
          start = 1;
      }
      for (int j = start; j < predictions.length; ++j)
        assertEquals("mismatch at row " + (rid) + ", p = " + j + ": " + outputs[j] + " != " + predictions[j] + ", predictions = " + Arrays.toString(predictions) + ", output = " + Arrays.toString(outputs), outputs[j], predictions[j], 1e-6);
    }
    @Override public void map(Chunk [] chks) {
      int nout = _m._parms._family == Family.multinomial ? _m._output.nclasses() + 1 : _m._parms._family == Family.binomial ? 3 : 1;
      Chunk[] outputChks = Arrays.copyOfRange(chks, chks.length - nout, chks.length);
      chks = Arrays.copyOf(chks, chks.length - nout);
      Chunk off = new C0DChunk(0, chks[0]._len);
      double[] tmp = new double[_m._output._dinfo._cats + _m._output._dinfo._nums];
      double[] predictions = new double[nout];
      double[] outputs = new double[nout];
      if (_offset) {
        off = chks[chks.length - 1];
        chks = Arrays.copyOf(chks, chks.length - 1);
      }
      if (_weights) {
        chks = Arrays.copyOf(chks, chks.length - 1);
      }
      for (int i = 0; i < chks[0]._len; ++i) {
        if (_weights || _offset)
          _m.score0(chks, off.atd(i), i, tmp, predictions);
        else
          _m.score0(chks, i, tmp, predictions);
        for (int j = 0; j < predictions.length; ++j)
          outputs[j] = outputChks[j].atd(i);
        checkScore(i + chks[0].start(), predictions, outputs);
      }
    }
  }

  @Test
  public void testStandardizedCoeff() {
    // test for multinomial
    testCoeffs(Family.multinomial, "smalldata/glm_test/multinomial_10_classes_10_cols_10000_Rows_train.csv", "C11");
    // test for binomial
    testCoeffs(Family.binomial, "smalldata/glm_test/binomial_20_cols_10KRows.csv", "C21");
    // test for Gaussian
    testCoeffs(Family.gaussian, "smalldata/glm_test/gaussian_20cols_10000Rows.csv", "C21");
  }

  private void testCoeffs(Family family, String fileName, String responseColumn) {
    try {
      Scope.enter();
      Frame train = parseTestFile(fileName);
      // set cat columns
      int numCols = train.numCols();
      int enumCols = (numCols-1)/2;
      for (int cindex=0; cindex<enumCols; cindex++) {
        train.replace(cindex, train.vec(cindex).toCategoricalVec()).remove();
      }
      int response_index = numCols-1;
      if (family.equals(Family.binomial) || (family.equals(Family.multinomial))) {
        train.replace((response_index), train.vec(response_index).toCategoricalVec()).remove();
      }
      DKV.put(train);
      Scope.track(train);

      GLMParameters params = new GLMParameters(family);
      params._standardize=true;
      params._response_column = responseColumn;
      params._train = train._key;
      GLMModel glm = new GLM(params).trainModel().get();
      Scope.track_generic(glm);

      // standardize numerical columns of train
      int numStart = enumCols;  // start of numerical columns
      int[] numCols2Transform = new int[enumCols];
      double[] colMeans = new double[enumCols];
      double[] oneOSigma = new double[enumCols];
      int countIndex = 0;
      HashMap<String, Double> cMeans = new HashMap<>();
      HashMap<String, Double> cSigmas = new HashMap<>();
      String[] cnames = train.names();
      for (int cindex = numStart; cindex < response_index; cindex++) {
        numCols2Transform[countIndex]=cindex;
        colMeans[countIndex] = train.vec(cindex).mean();
        oneOSigma[countIndex] = 1.0/train.vec(cindex).sigma();
        cMeans.put(cnames[cindex], colMeans[countIndex]);
        cSigmas.put(cnames[cindex], train.vec(cindex).sigma());
        countIndex++;
      }

      params._standardize = false;  // build a model on non-standardized columns with no standardization.
      GLMModel glmF = new GLM(params).trainModel().get();
      Scope.track_generic(glmF);

      HashMap<String, Double> coeffSF = glmF.coefficients(true);
      HashMap<String, Double> coeffF = glmF.coefficients();
      if (family.equals(Family.multinomial)) {
        double[] interPClass = new double[glmF._output._nclasses];
        for (String key : coeffSF.keySet()) {
          double temp1 = coeffSF.get(key);
          double temp2 = coeffF.get(key);
          if (Math.abs(temp1 - temp2) > 1e-6) { // coefficient same for categoricals, different for numericals
            String[] coNames = key.split("_");
            if (!(coNames[0].equals("Intercept"))) {  // skip over intercepts
              String colnames = coNames[0];
              interPClass[Integer.valueOf(coNames[1])] += temp2 * cMeans.get(colnames);
              temp2 = temp2 * cSigmas.get(colnames);
              assert Math.abs(temp1 - temp2) < 1e-6 : "Expected coefficients for " + coNames[0] + " is " + temp1 + " but actual " + temp2;
            }
          }
        }
        // check for equality of intercepts
        for (int index = 0; index < glmF._output._nclasses; index++) {
          String interceptKey = "Intercept_" + index;
          double temp1 = coeffSF.get(interceptKey);
          double temp2 = coeffF.get(interceptKey) + interPClass[index];
          assert Math.abs(temp1 - temp2) < 1e-6 : "Expected coefficients for " + interceptKey + " is " + temp1 + " but actual "
                  + temp2;
        }
      } else {
        double interceptOffset = 0;
        for (String key:coeffF.keySet()) {
          double temp1 = coeffSF.get(key);
          double temp2 = coeffF.get(key);
          if (Math.abs(temp1 - temp2) > 1e-6) {
            if (!key.equals("Intercept")) {
              interceptOffset += temp2*cMeans.get(key);
              temp2 = temp2*cSigmas.get(key);
              assert Math.abs(temp1 - temp2) < 1e-6 : "Expected coefficients for " + key + " is " + temp1 + " but actual " + temp2;
            }
          }
        }
        // check intercept terms
        double temp1 = coeffSF.get("Intercept");
        double temp2 = coeffF.get("Intercept")+interceptOffset;
        assert Math.abs(temp1 - temp2) < 1e-6 : "Expected coefficients for Intercept is " + temp1 + " but actual "
                + temp2;
      }
      new TestUtil.StandardizeColumns(numCols2Transform, colMeans, oneOSigma, train).doAll(train);
      DKV.put(train);
      Scope.track(train);

      params._standardize=false;
      params._train = train._key;
      GLMModel glmS = new GLM(params).trainModel().get();
      Scope.track_generic(glmS);

      if (family.equals(Family.multinomial)) {
        double[][] coeff1 = glm._output.getNormBetaMultinomial();
        double[][] coeff2 = glmS._output.getNormBetaMultinomial();
        for (int classind = 0; classind < coeff1.length; classind++) {
          assert TestUtil.equalTwoArrays(coeff1[classind], coeff2[classind], 1e-6);
        }
      } else {
        assert TestUtil.equalTwoArrays(glm._output.getNormBeta(), glmS._output.getNormBeta(), 1e-6);
      }
      HashMap<String, Double> coeff1 = glm.coefficients(true);
      HashMap<String, Double> coeff2 = glmS.coefficients(true);
      assert TestUtil.equalTwoHashMaps(coeff1, coeff2, 1e-6);
    } finally {
      Scope.exit();
    }
  }
  //------------------- simple tests on synthetic data------------------------------------
  @Test
  public void testGaussianRegression() throws InterruptedException, ExecutionException {
    Key raw = Key.make("gaussian_test_data_raw");
    Key parsed = Key.make("gaussian_test_data_parsed");
    GLMModel model = null;
    Frame fr = null, res = null;
    for (Family family : new Family[]{Family.gaussian, Family.AUTO}) {
      try {
        // make data so that the expected coefficients is icept = col[0] = 1.0
        FVecFactory.makeByteVec(raw, "x,y\n0,0\n1,0.1\n2,0.2\n3,0.3\n4,0.4\n5,0.5\n6,0.6\n7,0.7\n8,0.8\n9,0.9");
        fr = ParseDataset.parse(parsed, raw);
        GLMParameters params = new GLMParameters(family);
        params._train = fr._key;
        // params._response = 1;
        params._response_column = fr._names[1];
        params._lambda = new double[]{0};
  //      params._standardize= false;
        model = new GLM(params).trainModel().get();
        HashMap<String, Double> coefs = model.coefficients();
        assertEquals(0.0, coefs.get("Intercept"), 1e-4);
        assertEquals(0.1, coefs.get("x"), 1e-4);
        testScoring(model,fr);
      } finally {
        if (fr != null) fr.remove();
        if (res != null) res.remove();
        if (model != null) model.remove();
      }
    }
  }

  /**
   * Test Poisson regression on simple and small synthetic dataset.
   * Equation is: y = exp(x+1);
   */
  @Test
  public void testPoissonRegression() throws InterruptedException, ExecutionException {
    Key raw = Key.make("poisson_test_data_raw");
    Key parsed = Key.make("poisson_test_data_parsed");

    GLMModel model = null;
    Frame fr = null, res = null;
    try {
      // make data so that the expected coefficients is icept = col[0] = 1.0
      FVecFactory.makeByteVec(raw, "x,y\n0,2\n1,4\n2,8\n3,16\n4,32\n5,64\n6,128\n7,256");
      fr = ParseDataset.parse(parsed, raw);
      Vec v = fr.vec(0);
      System.out.println(v.min() + ", " + v.max() + ", mean = " + v.mean());
      GLMParameters params = new GLMParameters(Family.poisson);
      params._train = fr._key;
      // params._response = 1;
      params._response_column = fr._names[1];
      params._lambda = new double[]{0};
      params._standardize = false;
      model = new GLM(params).trainModel().get();
      for (double c : model.beta())
        assertEquals(Math.log(2), c, 1e-2); // only 1e-2 precision cause the perfect solution is too perfect -> will trigger grid search
      testScoring(model,fr);
      model.delete();
      fr.delete();

      // Test 2, example from http://www.biostat.umn.edu/~dipankar/bmtry711.11/lecture_13.pdf
      FVecFactory.makeByteVec(raw, "x,y\n1,0\n2,1\n3,2\n4,3\n5,1\n6,4\n7,9\n8,18\n9,23\n10,31\n11,20\n12,25\n13,37\n14,45\n150,7.193936e+16\n");
      fr = ParseDataset.parse(parsed, raw);
      GLMParameters params2 = new GLMParameters(Family.poisson);
      params2._train = fr._key;
      // params2._response = 1;
      params2._response_column = fr._names[1];
      params2._lambda = new double[]{0};
      params2._standardize = true;
      params2._beta_epsilon = 1e-5;
      model = new GLM(params2).trainModel().get();
      assertEquals(0.3396, model.beta()[1], 1e-1);
      assertEquals(0.2565, model.beta()[0], 1e-1);
      // test scoring
      testScoring(model,fr);
    } finally {
      if (fr != null) fr.delete();
      if (res != null) res.delete();
      if (model != null) model.delete();
    }
  }


  /**
   * Test Gamma regression on simple and small synthetic dataset.
   * Equation is: y = 1/(x+1);
   *
   * @throws ExecutionException
   * @throws InterruptedException
   */
  @Test
  public void testGammaRegression() throws InterruptedException, ExecutionException {
    GLMModel model = null;
    Frame fr = null, res = null;
    try {
      // make data so that the expected coefficients is icept = col[0] = 1.0
      Key raw = Key.make("gamma_test_data_raw");
      Key parsed = Key.make("gamma_test_data_parsed");
      FVecFactory.makeByteVec(raw, "x,y\n0,1\n1,0.5\n2,0.3333333\n3,0.25\n4,0.2\n5,0.1666667\n6,0.1428571\n7,0.125");
      fr = ParseDataset.parse(parsed, raw);
//      /public GLM2(String desc, Key dest, Frame src, Family family, Link link, double alpha, double lambda) {
//      double [] vals = new double[] {1.0,1.0};
      //public GLM2(String desc, Key dest, Frame src, Family family, Link link, double alpha, double lambda) {
      GLMParameters params = new GLMParameters(Family.gamma);
      // params._response = 1;
      params._response_column = fr._names[1];
      params._train = parsed;
      params._lambda = new double[]{0};
      model = new GLM(params).trainModel().get();
      for (double c : model.beta()) assertEquals(1.0, c, 1e-4);
      // test scoring
      testScoring(model,fr);
    } finally {
      if (fr != null) fr.delete();
      if (res != null) res.delete();
      if (model != null) model.delete();
    }
  }

////  //simple tweedie test
//  @Test public void testTweedieRegression() throws InterruptedException, ExecutionException{
//    Key raw = Key.make("gaussian_test_data_raw");
//    Key parsed = Key.make("gaussian_test_data_parsed");
//    Key<GLMModel> modelKey = Key.make("gaussian_test");
//    Frame fr = null;
//    GLMModel model = null;
//    try {
//      // make data so that the expected coefficients is icept = col[0] = 1.0
//      FVecFactory.makeByteVec(raw, "x,y\n0,0\n1,0.1\n2,0.2\n3,0.3\n4,0.4\n5,0.5\n6,0.6\n7,0.7\n8,0.8\n9,0.9\n0,0\n1,0\n2,0\n3,0\n4,0\n5,0\n6,0\n7,0\n8,0\n9,0");
//      fr = ParseDataset.parse(parsed, new Key[]{raw});
//      double [] powers = new double [] {1.5,1.1,1.9};
//      double [] intercepts = new double []{3.643,1.318,9.154};
//      double [] xs = new double []{-0.260,-0.0284,-0.853};
//      for(int i = 0; i < powers.length; ++i){
//        DataInfo dinfo = new DataInfo(fr, 1, false, DataInfo.TransformType.NONE);
//        GLMParameters glm = new GLMParameters(Family.tweedie);
//
//        new GLM2("GLM test of gaussian(linear) regression.",Key.make(),modelKey,dinfo,glm,new double[]{0},0).fork().get();
//        model = DKV.get(modelKey).get();
//        testHTML(model);
//        HashMap<String, Double> coefs = model.coefficients();
//        assertEquals(intercepts[i],coefs.get("Intercept"),1e-3);
//        assertEquals(xs[i],coefs.get("x"),1e-3);
//      }
//    }finally{
//      if( fr != null ) fr.delete();
//      if(model != null)model.delete();
//    }
//  }


  @Test
  public void testAllNAs() {
    Key raw = Key.make("gamma_test_data_raw");
    Key parsed = Key.make("gamma_test_data_parsed");
    FVecFactory.makeByteVec(raw, "x,y,z\n1,0,NA\n2,NA,1\nNA,3,2\n4,3,NA\n5,NA,1\nNA,6,4\n7,NA,9\n8,NA,18\nNA,9,23\n10,31,NA\nNA,11,20\n12,NA,25\nNA,13,37\n14,45,NA\n");
    Frame fr = ParseDataset.parse(parsed, raw);
    GLM job = null;
    try {
      GLMParameters params = new GLMParameters(Family.poisson);
      // params._response = 1;
      params._response_column = fr._names[1];
      params._train = parsed;
      params._lambda = new double[]{0};
      params._missing_values_handling = MissingValuesHandling.Skip;
      GLM glm = new GLM( params);
      glm.trainModel().get();
      assertFalse("should've thrown IAE", true);
    } catch (IllegalArgumentException e) {
      assertTrue(e.getMessage(), e.getMessage().contains("No rows left in the dataset"));
    } finally {
      fr.delete();
    }
  }

  // Make sure all three implementations of ginfo computation in GLM get the same results
  @Test
  public void testGradientTask() {
    Key parsed = Key.make("cars_parsed");
    Frame fr = null;
    DataInfo dinfo = null;
    try {
      fr = parseTestFile(parsed, "smalldata/junit/mixcat_train.csv");
      GLMParameters params = new GLMParameters(Family.binomial, Family.binomial.defaultLink, new double[]{0}, new double[]{0}, 0, 0);
      // params._response = fr.find(params._response_column);
      params._train = parsed;
      params._lambda = new double[]{0};
      params._use_all_factor_levels = true;
      fr.add("Useless", fr.remove("Useless"));

      dinfo = new DataInfo(fr, null, 1, params._use_all_factor_levels || params._lambda_search, params._standardize ? DataInfo.TransformType.STANDARDIZE : DataInfo.TransformType.NONE, DataInfo.TransformType.NONE, true, false, false, false, false, false);
      DKV.put(dinfo._key,dinfo);
      double [] beta = MemoryManager.malloc8d(dinfo.fullN()+1);
      Random rnd = new Random(987654321);
      for (int i = 0; i < beta.length; ++i)
        beta[i] = 1 - 2 * rnd.nextDouble();

      GLMGradientTask grtSpc = new GLMBinomialGradientTask(null,dinfo, params, params._lambda[0], beta).doAll(dinfo._adaptedFrame);
      GLMGradientTask grtGen = new GLMGenericGradientTask(null,dinfo, params, params._lambda[0], beta).doAll(dinfo._adaptedFrame);
      for (int i = 0; i < beta.length; ++i)
        assertEquals("gradients differ", grtSpc._gradient[i], grtGen._gradient[i], 1e-4);
      params = new GLMParameters(Family.gaussian, Family.gaussian.defaultLink, new double[]{0}, new double[]{0}, 0, 0);
      params._use_all_factor_levels = false;
      dinfo.remove();
      dinfo = new DataInfo(fr, null, 1, params._use_all_factor_levels || params._lambda_search, params._standardize ? DataInfo.TransformType.STANDARDIZE : DataInfo.TransformType.NONE, DataInfo.TransformType.NONE, true, false, false, false, false, false);
      DKV.put(dinfo._key,dinfo);
      beta = MemoryManager.malloc8d(dinfo.fullN()+1);
      rnd = new Random(1987654321);
      for (int i = 0; i < beta.length; ++i)
        beta[i] = 1 - 2 * rnd.nextDouble();
      grtSpc = new GLMGaussianGradientTask(null,dinfo, params, params._lambda[0], beta).doAll(dinfo._adaptedFrame);
      grtGen = new GLMGenericGradientTask(null,dinfo, params, params._lambda[0], beta).doAll(dinfo._adaptedFrame);
      for (int i = 0; i < beta.length; ++i)
        assertEquals("gradients differ: " + Arrays.toString(grtSpc._gradient) + " != " + Arrays.toString(grtGen._gradient), grtSpc._gradient[i], grtGen._gradient[i], 1e-4);
      dinfo.remove();
    } finally {
      if (fr != null) fr.delete();
      if (dinfo != null) dinfo.remove();
    }
  }


 */
