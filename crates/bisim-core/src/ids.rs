use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

/// Dense index of a state inside one transition system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StateId(pub u32);

impl StateId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s{}", self.0)
    }
}

/// Dense index of an interned label.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LabelId(pub u32);

impl LabelId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Bidirectional map between values and dense `u32` ids.
#[derive(Debug, Clone)]
pub struct Interner<T> {
    values: Vec<T>,
    ids: HashMap<T, u32>,
}

impl<T> Default for Interner<T> {
    fn default() -> Self {
        Self {
            values: Vec::new(),
            ids: HashMap::new(),
        }
    }
}

impl<T: Eq + Hash + Clone> Interner<T> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn intern(&mut self, value: T) -> u32 {
        if let Some(&id) = self.ids.get(&value) {
            return id;
        }
        let id = u32::try_from(self.values.len()).expect("interner overflow");
        self.values.push(value.clone());
        self.ids.insert(value, id);
        id
    }

    pub fn get(&self, value: &T) -> Option<u32> {
        self.ids.get(value).copied()
    }

    pub fn resolve(&self, id: u32) -> &T {
        &self.values[id as usize]
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn values(&self) -> &[T] {
        &self.values
    }
}
